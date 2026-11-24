"""steplog: run a multi-step task against a UI you cannot make idempotent.

A step is three callables:
    observe() -> JSON value   read the screen before acting
    act()                     do the side effect (may raise ActionError)
    present(before) -> bool   read the screen: did my side effect happen?

The runner writes an intent record (with the observation) before acting and a
done record after. After any error, and when resuming after a crash, it reads
the screen before it acts again. If it cannot tell, it stops and escalates.

By default one successful read is believed. Pass a verifier from verify.py
(for example verify.Sequential) when reads can be wrong, as they can when a
model reads a screenshot. The verifier decides how many reads to spend and may
answer "unsure", which escalates.
"""
import json
import os

__all__ = ["Step", "Journal", "Runner", "NaiveRunner", "DoneOnlyRunner",
           "ActionError", "ReadError", "JournalCorrupt",
           "DONE", "ESCALATED", "FAILED"]

DONE, ESCALATED, FAILED = "done", "needs_human", "failed"


class ActionError(Exception):
    """The action reported a failure. The side effect may or may not exist."""


class ReadError(Exception):
    """A read of the screen failed. Nothing was changed."""


class JournalCorrupt(Exception):
    pass


class Step:
    __slots__ = ("name", "act", "observe", "present")

    def __init__(self, name, act, observe, present):
        self.name, self.act, self.observe, self.present = name, act, observe, present


class Journal:
    """Append-only JSONL file. A torn final line (crash mid-write) is ignored
    on load. A bad line anywhere else raises JournalCorrupt."""

    def __init__(self, path, fsync=True):
        self.path, self.fsync = path, fsync
        self.records = []
        if os.path.exists(path):
            with open(path, "rb") as f:
                lines = f.read().split(b"\n")
            good = 0
            for n, raw in enumerate(lines):
                if not raw.strip():
                    good += len(raw) + 1
                    continue
                try:
                    self.records.append(json.loads(raw))
                except ValueError:
                    if any(x.strip() for x in lines[n + 1:]):
                        raise JournalCorrupt("%s line %d" % (path, n + 1))
                    # torn tail: cut it off so the next append starts clean
                    with open(path, "r+b") as f:
                        f.truncate(good)
                    break
                good += len(raw) + 1
        self._f = open(path, "ab")

    def append(self, rec):
        self._f.write(json.dumps(rec, separators=(",", ":")).encode() + b"\n")
        self._f.flush()
        if self.fsync:
            os.fsync(self._f.fileno())
        self.records.append(rec)

    def close(self):
        self._f.close()

    def state(self, task):
        """-> (done step indexes, {index: observation} of open intents, escalated?)"""
        done, intents, esc = set(), {}, False
        for r in self.records:
            if r.get("t") != task:
                continue
            k = r["k"]
            if k == "intent":
                intents[r["i"]] = r.get("before")
            elif k == "done":
                done.add(r["i"])
                intents.pop(r["i"], None)
            elif k == "escalated":
                esc = True
        return done, intents, esc


class Runner:
    """Intent record, act, verify, done record.

    prior_ok / prior_err / prior_restart are the probabilities that the change
    is on the screen after an action that reported success, after one that
    raised ActionError, and when an open intent is found after a restart. Only
    a verifier that uses priors (verify.Sequential) looks at them.
    """

    def __init__(self, journal, max_attempts=4, read_attempts=3, verify_after=True,
                 verifier=None, prior_ok=0.98, prior_err=0.5, prior_restart=0.5):
        self.j = journal
        self.max_attempts = max_attempts
        self.read_attempts = read_attempts
        self.verify_after = verify_after
        self.verifier = verifier
        self.prior_ok, self.prior_err, self.prior_restart = prior_ok, prior_err, prior_restart
        self.stats = {"recovered": 0, "redone": 0, "noop_caught": 0}

    def _read(self, fn, *a):
        for _ in range(self.read_attempts):
            try:
                return True, fn(*a)
            except ReadError:
                pass
        return False, None

    def _observe(self, s):
        """-> (ok, observation)"""
        if self.verifier is None:
            return self._read(s.observe)
        ok, value, _ = self.verifier.observe(s.observe)
        return ok, value

    def _check(self, s, before, prior):
        """-> (True | False | None for cannot tell, evidence dict for the journal)"""
        if self.verifier is None:
            ok, there = self._read(s.present, before)
            return (bool(there) if ok else None), {}
        v = self.verifier.decide(lambda: s.present(before), prior)
        ev = {"reads": v.reads, "yes": v.yes, "no": v.no}
        if v.p is not None:
            ev["p"] = v.p
        return {"present": True, "absent": False}.get(v.answer), ev

    def _escalate(self, task, i, why, ev=None):
        self.j.append(dict({"t": task, "i": i, "k": "escalated", "why": why}, **(ev or {})))
        return ESCALATED

    def run(self, task, steps):
        done, intents, esc = self.j.state(task)
        if esc:
            return ESCALATED
        for i, s in enumerate(steps):
            if i in done:
                continue
            if i in intents:
                # We crashed somewhere between writing the intent and writing
                # done. Look before touching anything.
                before = intents[i]
                there, ev = self._check(s, before, self.prior_restart)
                if there is None:
                    return self._escalate(task, i, "cannot verify after restart", ev)
                if there:
                    self.stats["recovered"] += 1
                    self.j.append(dict({"t": task, "i": i, "k": "done", "how": "found"}, **ev))
                    continue
                self.stats["redone"] += 1
            else:
                ok, before = self._observe(s)
                if not ok:
                    return self._escalate(task, i, "cannot observe")
                self.j.append({"t": task, "i": i, "k": "intent", "n": s.name,
                               "before": before})
            how, ev = None, {}
            for _ in range(self.max_attempts):
                try:
                    s.act()
                    failed = False
                except ActionError:
                    failed = True
                if failed or self.verify_after:
                    there, ev = self._check(s, before,
                                            self.prior_err if failed else self.prior_ok)
                    if there is None:
                        return self._escalate(task, i, "cannot verify after act", ev)
                    if there:
                        how = "found" if failed else "acted"
                        if failed:
                            self.stats["recovered"] += 1
                        break
                    if not failed:
                        self.stats["noop_caught"] += 1
                else:
                    how = "acted"
                    break
            if how is None:
                return self._escalate(task, i, "attempts exhausted")
            self.j.append(dict({"t": task, "i": i, "k": "done", "how": how}, **ev))
        return DONE


class DoneOnlyRunner:
    """Baseline: checkpoint after each step, retry on error, never read."""

    def __init__(self, journal, max_attempts=4):
        self.j, self.max_attempts = journal, max_attempts

    def run(self, task, steps):
        done, _, _ = self.j.state(task)
        for i, s in enumerate(steps):
            if i in done:
                continue
            for _ in range(self.max_attempts):
                try:
                    s.act()
                    break
                except ActionError:
                    pass
            else:
                return FAILED
            self.j.append({"t": task, "i": i, "k": "done", "how": "acted"})
        return DONE


class NaiveRunner:
    """Baseline: no journal. Retry on error; a restart begins from step 0."""

    def __init__(self, journal=None, max_attempts=4):
        self.max_attempts = max_attempts

    def run(self, task, steps):
        for s in steps:
            for _ in range(self.max_attempts):
                try:
                    s.act()
                    break
                except ActionError:
                    pass
            else:
                return FAILED
        return DONE
