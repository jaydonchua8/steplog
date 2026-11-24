"""verify: decide "is my change on the screen?" when a read can be wrong.

steplog 1 trusted every successful read. That is not safe when the reader is a
model looking at a screenshot. A read that wrongly says "present" makes the
runner record a step that never happened. A read that wrongly says "absent"
makes it click again and charge twice.

A verifier takes a function that performs one read and a prior probability
that the change is there, and returns a Verdict. Three are defined here:

    Single       one successful read is believed (the steplog 1 behaviour)
    MajorityOf   k reads, majority wins, a tie is "unsure"
    Sequential   Bayesian sequential test: keep reading until the posterior
                 probability is above 1 - alpha (present) or below beta
                 (absent), or the read budget runs out ("unsure")

"Unsure" is a first-class answer. The runner turns it into an escalation
instead of guessing.
"""
import math

from steplog import ReadError

__all__ = ["Verdict", "Single", "MajorityOf", "Sequential", "estimate_rates",
           "estimate_repeat",
           "PRESENT", "ABSENT", "UNSURE"]

PRESENT, ABSENT, UNSURE = "present", "absent", "unsure"


class Verdict:
    __slots__ = ("answer", "reads", "yes", "no", "p")

    def __init__(self, answer, reads, yes, no, p=None):
        self.answer, self.reads, self.yes, self.no, self.p = answer, reads, yes, no, p

    def __repr__(self):
        return "Verdict(%s, reads=%d, yes=%d, no=%d, p=%r)" % (
            self.answer, self.reads, self.yes, self.no, self.p)


def _logit(p):
    return math.log(p / (1.0 - p))


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _agree(read, lead, budget):
    """Read a value (any JSON scalar) until the most common answer leads the
    runner-up by `lead` reads. -> (ok, value, reads used). Used for the
    observation taken before acting, which is not a yes/no question. A wrong
    observation is worse than a wrong check: every later check of that step
    compares against it."""
    seen, vals, used = {}, {}, 0
    while used < budget:
        used += 1
        try:
            v = read()
        except ReadError:
            continue
        k = repr(v)
        seen[k] = seen.get(k, 0) + 1
        vals[k] = v
        ranked = sorted(seen.values(), reverse=True)
        if ranked[0] - (ranked[1] if len(ranked) > 1 else 0) >= lead:
            return True, vals[max(seen, key=seen.get)], used
    return False, None, used


class Single:
    """Believe the first read that does not fail."""

    def __init__(self, read_attempts=3):
        self.read_attempts = read_attempts

    def observe(self, read):
        return _agree(read, 1, self.read_attempts)

    def decide(self, read, prior=0.5):
        for n in range(1, self.read_attempts + 1):
            try:
                v = bool(read())
            except ReadError:
                continue
            return Verdict(PRESENT if v else ABSENT, n, int(v), int(not v))
        return Verdict(UNSURE, self.read_attempts, 0, 0)


class MajorityOf:
    """k successful reads, majority wins. Ignores the prior. Stops early once
    one side cannot be caught."""

    def __init__(self, k=3, max_reads=None):
        if k < 1 or k % 2 == 0:
            raise ValueError("k must be odd and positive")
        self.k = k
        self.max_reads = max_reads or 2 * k

    def observe(self, read):
        return _agree(read, self.k // 2 + 1, self.max_reads)

    def decide(self, read, prior=0.5):
        yes = no = used = 0
        need = self.k // 2 + 1
        while used < self.max_reads and yes < need and no < need:
            used += 1
            try:
                v = bool(read())
            except ReadError:
                continue
            yes += v
            no += not v
        if yes >= need:
            return Verdict(PRESENT, used, yes, no)
        if no >= need:
            return Verdict(ABSENT, used, yes, no)
        return Verdict(UNSURE, used, yes, no)


class Sequential:
    """Bayesian sequential test with asymmetric stopping rules.

    e_fp   P(a read says present | the change is absent)
    e_fn   P(a read says absent  | the change is present)
    alpha  stop and answer "present" when P(absent | reads) <= alpha.
           This bounds steps recorded as done that did not happen.
    beta   stop and answer "absent" when P(present | reads) <= beta.
           This bounds repeated clicks, which is what double charges a guest.
    max_reads  budget, counting failed reads. Out of budget -> "unsure".
    repeat     P(a wrong read is followed by another wrong read). None means
           reads are treated as independent. With a value, a read that agrees
           with the previous one counts for much less, because a stale or
           confusing screen tends to be misread the same way twice.

    The bounds hold only if the rates are right and the error model fits.
    bench.py measures what happens when neither is true.
    """

    def __init__(self, e_fp, e_fn, alpha=1e-3, beta=1e-3, max_reads=8, repeat=None):
        for name, v in (("e_fp", e_fp), ("e_fn", e_fn)):
            if not 0.0 < v < 0.5:
                raise ValueError("%s must be in (0, 0.5), got %r" % (name, v))
        for name, v in (("alpha", alpha), ("beta", beta)):
            if not 0.0 < v < 0.5:
                raise ValueError("%s must be in (0, 0.5), got %r" % (name, v))
        if max_reads < 1:
            raise ValueError("max_reads must be at least 1")
        if repeat is not None and not 0.0 <= repeat < 1.0:
            raise ValueError("repeat must be in [0, 1), got %r" % (repeat,))
        self.e_fp, self.e_fn = e_fp, e_fn
        self.alpha, self.beta, self.max_reads = alpha, beta, max_reads
        self.repeat = repeat
        # after a wrong read, the chance the next one is wrong too
        self.again_fp = e_fp if repeat is None else max(repeat, e_fp)
        self.again_fn = e_fn if repeat is None else max(repeat, e_fn)
        self.hi = _logit(1.0 - alpha)
        self.lo = _logit(beta)
        # Lead needed so that a wrong observation is rarer than min(alpha,
        # beta) if reads are independent: (e / (1 - e)) ** lead <= target.
        e = max(e_fp, e_fn)
        self.lead = min(max_reads, max(1, int(math.ceil(
            math.log(min(alpha, beta)) / math.log(e / (1.0 - e))))))

    def observe(self, read):
        return _agree(read, self.lead, self.max_reads)

    def decide(self, read, prior=0.5):
        # A prior is a guess about the action, not evidence from the screen.
        # Keep it inside the stopping bounds so one contrary read always
        # reopens the question.
        prior = min(max(prior, 0.01), 0.99)
        x = min(max(_logit(prior), self.lo), self.hi)
        yes = no = used = 0
        prev = None
        # At least one successful read is required, however strong the prior.
        while used < self.max_reads and (yes + no == 0 or self.lo < x < self.hi):
            used += 1
            try:
                v = bool(read())
            except ReadError:
                continue
            # If the change is present, a "no" is a wrong read; if it is
            # absent, a "yes" is. So the previous read says which hypothesis
            # is currently in a run of wrong reads.
            p_no = self.again_fn if prev is False else self.e_fn       # given present
            p_yes = self.again_fp if prev is True else self.e_fp       # given absent
            if v:
                yes += 1
                x += math.log((1.0 - p_no) / p_yes)
            else:
                no += 1
                x += math.log(p_no / (1.0 - p_yes))
            prev = v
        p = round(_sigmoid(x), 6)
        if yes + no and x >= self.hi:
            return Verdict(PRESENT, used, yes, no, p)
        if yes + no and x <= self.lo:
            return Verdict(ABSENT, used, yes, no, p)
        return Verdict(UNSURE, used, yes, no, p)


def estimate_rates(labelled, floor=0.005):
    """labelled: iterable of (truth, read) booleans, e.g. screenshots a person
    checked. -> (e_fp, e_fn) with add-one smoothing, clamped to [floor, 0.45]
    so that a small clean sample cannot claim a reader is perfect."""
    fp = fn = pos = neg = 0
    for truth, read in labelled:
        if truth:
            pos += 1
            fn += not read
        else:
            neg += 1
            fp += bool(read)
    if not pos or not neg:
        raise ValueError("need labelled reads of both present and absent screens")
    clamp = lambda v: min(max(v, floor), 0.45)
    return clamp((fp + 1.0) / (neg + 2.0)), clamp((fn + 1.0) / (pos + 2.0))


def estimate_repeat(pairs):
    """pairs: iterable of (truth, first read, second read) for two looks at
    the same screen. -> P(second read wrong | first read wrong), with add-one
    smoothing. Raises if no first read was wrong."""
    again = wrong = 0
    for truth, r1, r2 in pairs:
        if bool(r1) != bool(truth):
            wrong += 1
            again += bool(r2) != bool(truth)
    if not wrong:
        raise ValueError("no wrong first reads in the sample")
    return (again + 1.0) / (wrong + 2.0)
