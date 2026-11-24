"""A fake hotel property management system with the failure modes of a UI
driven by clicks: no idempotency keys, actions that time out after they were
applied, clicks that silently do nothing, reads that fail, reads that succeed
with the wrong answer, and a worker that can die at any point. Also three task templates built on steplog.Step.

Rows carry a hidden `_src` field (which task and step wrote them). The agent
never sees it. It exists so the benchmark can count duplicates exactly.
"""
import hashlib
import json
import random

from steplog import Step, ActionError, ReadError


class Crash(BaseException):
    """The worker process died. Not an Exception on purpose: runners must not
    be able to catch it."""


class Faults:
    """fp / fn: a read that succeeds but is wrong, the way a model can misread
    a screenshot. fp is P(sees my row | it is not there), fn is P(does not see
    it | it is there). burst is the chance that a wrong read of the same thing
    is wrong again on the next look (a stale or confusing screen), which makes
    repeated reads correlated instead of independent."""

    def __init__(self, seed=0, fail_before=0.0, ack_lost=0.0, noop=0.0,
                 crash=0.0, read_fail=0.0, fp=0.0, fn=0.0, burst=0.0):
        self.rng = random.Random(seed)
        self.fail_before, self.ack_lost, self.noop = fail_before, ack_lost, noop
        self.crash, self.read_fail = crash, read_fail
        self.fp, self.fn, self.burst = fp, fn, burst
        self._wrong = {}          # read key -> truth it was last misread at

    def misread(self, key, truth):
        """-> True if this read of `key` should come back wrong. Draws nothing
        when fp = fn = 0, so runs without misreads keep their random stream."""
        if not (self.fp or self.fn):
            return False
        if self.burst and self._wrong.get(key, self) == truth:
            wrong = self.rng.random() < self.burst
        else:
            wrong = self.rng.random() < (self.fn if truth else self.fp)
        if wrong:
            self._wrong[key] = truth
        else:
            self._wrong.pop(key, None)
        return wrong

    def action(self):
        r = self.rng.random()
        for name, p in (("crash_before", self.crash / 2), ("crash_after", self.crash / 2),
                        ("fail_before", self.fail_before), ("ack_lost", self.ack_lost),
                        ("noop", self.noop)):
            if r < p:
                return name
            r -= p
        return "ok"

    def read(self):
        return self.rng.random() < self.read_fail


class FakePMS:
    APPEND = ("charges", "tickets", "sms")

    def __init__(self, faults=None):
        self.f = faults or Faults()
        self.charges, self.tickets, self.sms = [], [], []
        self.fields = {}          # (reservation, field) -> value
        self.acts = self.reads = self.ops = 0
        self.scheduled = []       # (op number, table, row) written by a human clerk

    # ---- persistence, used by demo.py to survive a real process kill
    def save(self, path):
        d = {"charges": self.charges, "tickets": self.tickets, "sms": self.sms,
             "fields": [[k[0], k[1], v] for k, v in self.fields.items()]}
        with open(path, "w") as f:
            json.dump(d, f)

    @classmethod
    def load(cls, path, faults=None):
        p = cls(faults)
        with open(path) as f:
            d = json.load(f)
        p.charges, p.tickets, p.sms = d["charges"], d["tickets"], d["sms"]
        p.fields = {(a, b): v for a, b, v in d["fields"]}
        return p

    # ---- internals
    def _tick(self):
        self.ops += 1
        while self.scheduled and self.scheduled[0][0] <= self.ops:
            _, table, row = self.scheduled.pop(0)
            getattr(self, table).append(dict(row, _src="clerk"))

    def _act(self, apply):
        self._tick()
        self.acts += 1
        o = self.f.action()
        if o == "crash_before":
            raise Crash()
        if o == "fail_before":
            raise ActionError("element not found")
        if o != "noop":
            apply()
        if o == "crash_after":
            raise Crash()
        if o == "ack_lost":
            raise ActionError("timed out waiting for confirmation")

    def _read(self):
        self._tick()
        self.reads += 1
        if self.f.read():
            raise ReadError("screen did not load")

    # ---- actions (what the agent clicks through)
    def append(self, table, row, src=None):
        self._act(lambda: getattr(self, table).append(dict(row, _src=src)))

    def set_field(self, res, field, value):
        self._act(lambda: self.fields.__setitem__((res, field), value))

    def _wrong(self, key, truth):
        m = getattr(self.f, "misread", None)
        return bool(m and m(key, truth))

    # ---- reads (what the agent sees). Each can fail, and each can succeed
    # with a wrong answer if the fault model has fp / fn set.
    def count(self, table, row):
        self._read()
        n = sum(1 for r in getattr(self, table)
                if all(r.get(k) == v for k, v in row.items()))
        key = ("count", table, tuple(sorted(row.items())))
        if n and self._wrong(key + (True,), True):
            return n - 1                    # overlooked one row
        if self._wrong(key + (False,), False):
            return n + 1                    # saw a row that is not there
        return n

    def has_tag(self, table, key, tag):
        self._read()
        truth = any(tag in r.get(key, "") for r in getattr(self, table))
        return truth != self._wrong(("tag", table, tag), truth)

    def field_is(self, res, field, value):
        self._read()
        truth = self.fields.get((res, field)) == value
        return truth != self._wrong(("field", res, field, value), truth)

    def get_field(self, res, field):
        self._read()
        return self.fields.get((res, field))


def tag_for(task, i):
    return "ref " + hashlib.sha1(("%s/%d" % (task, i)).encode()).hexdigest()[:8]


def append_step(pms, task, i, table, row, text_key, mode):
    """mode 'count': remember how many identical rows exist, later ask whether
    there is one more. mode 'tag': put a short reference in a free-text field
    and later search for it."""
    src = [task, i]
    if mode == "tag":
        tag = tag_for(task, i)
        row = dict(row, **{text_key: row[text_key] + " (" + tag + ")"})
        return Step("%s+" % table, lambda: pms.append(table, row, src),
                    lambda: None, lambda before: pms.has_tag(table, text_key, tag))
    return Step("%s+" % table, lambda: pms.append(table, row, src),
                lambda: pms.count(table, row),
                lambda before: pms.count(table, row) > before)


def set_step(pms, res, field, value):
    return Step("set %s" % field, lambda: pms.set_field(res, field, value),
                lambda: None, lambda before: pms.field_is(res, field, value))


TEMPLATES = ("late_checkout", "room_move", "service_request")


def build(pms, task, template, res, room, mode="count"):
    """-> list of Step. SMS is always verified by count: I do not want a
    reference code in a text the guest reads."""
    A = lambda i, table, row, key, m=mode: append_step(pms, task, i, table, row, key, m)
    if template == "late_checkout":
        return [
            A(0, "charges", {"res": res, "amount": 40, "desc": "Late checkout fee"}, "desc"),
            set_step(pms, res, "departure_time", "14:00"),
            A(2, "tickets", {"room": room, "text": "Late checkout, clean after 14:00"}, "text"),
            A(3, "sms", {"res": res, "text": "Your 2pm late checkout is confirmed."}, "text", "count"),
        ]
    if template == "room_move":
        new = room + 100
        return [
            set_step(pms, res, "room", new),
            A(1, "charges", {"res": res, "amount": 25, "desc": "Room upgrade"}, "desc"),
            A(2, "tickets", {"room": room, "text": "Guest moved out, full clean"}, "text"),
            A(3, "tickets", {"room": new, "text": "Deliver luggage from %d" % room}, "text"),
            A(4, "sms", {"res": res, "text": "Your new room %d is ready." % new}, "text", "count"),
        ]
    if template == "service_request":
        return [
            A(0, "tickets", {"room": room, "text": "Extra towels"}, "text"),
            A(1, "sms", {"res": res, "text": "Towels are on the way."}, "text", "count"),
        ]
    raise ValueError(template)


# which (step index -> table) rows a clerk might also enter by hand
CLERK_ROWS = {
    "late_checkout": [(2, "tickets", lambda res, room: {"room": room, "text": "Late checkout, clean after 14:00"})],
    "room_move": [(2, "tickets", lambda res, room: {"room": room, "text": "Guest moved out, full clean"})],
    "service_request": [(0, "tickets", lambda res, room: {"room": room, "text": "Extra towels"})],
}


def audit(pms, task, template, res, room):
    """Ground truth for one task -> (duplicate side effects, missing side effects)."""
    steps = build(FakePMS(), task, template, res, room)
    dup = missing = 0
    rows = pms.charges + pms.tickets + pms.sms
    for i, s in enumerate(steps):
        if s.name.startswith("set "):
            continue
        n = sum(1 for r in rows if r.get("_src") == [task, i])
        dup += max(0, n - 1)
        missing += n == 0
    want = {"late_checkout": [("departure_time", "14:00")],
            "room_move": [("room", room + 100)], "service_request": []}[template]
    for field, value in want:
        missing += pms.fields.get((res, field)) != value
    return dup, missing
