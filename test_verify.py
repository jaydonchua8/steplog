"""Tests for verify.py, the noisy-read fault model, and report.py."""
import os
import random
import tempfile
import unittest

from steplog import Journal, Runner, Step, ActionError, ReadError, DONE, ESCALATED
from verify import (Single, MajorityOf, Sequential, estimate_rates, estimate_repeat,
                    PRESENT, ABSENT, UNSURE)
from fakepms import FakePMS, Faults, Crash, build, audit
import report


def reader(answers):
    """A read function that returns scripted answers. An Exception instance in
    the list is raised."""
    it = iter(answers)

    def read():
        v = next(it)
        if isinstance(v, Exception):
            raise v
        return v
    return read


def load(path):
    j = Journal(path)
    j.close()
    return j.records


def noisy(rng, truth, e_fp, e_fn):
    return lambda: (rng.random() >= e_fn) if truth else (rng.random() < e_fp)


class TestSequential(unittest.TestCase):
    def test_strong_prior_needs_one_agreeing_read(self):
        v = Sequential(0.02, 0.06).decide(reader([True]), prior=0.98)
        self.assertEqual((v.answer, v.reads), (PRESENT, 1))

    def test_even_prior_needs_more_than_one_read(self):
        v = Sequential(0.02, 0.06).decide(reader([True, True, True]), prior=0.5)
        self.assertEqual((v.answer, v.reads), (PRESENT, 2))

    def test_a_certain_prior_still_requires_a_read(self):
        s = Sequential(0.02, 0.06)
        self.assertEqual(s.decide(reader([False] * 8), prior=1.0).answer, ABSENT)
        v = s.decide(reader([ReadError()] * 8), prior=1.0)
        self.assertEqual((v.answer, v.yes + v.no), (UNSURE, 0))

    def test_conflicting_reads_end_unsure_not_guessed(self):
        v = Sequential(0.05, 0.05, max_reads=6).decide(reader([True, False] * 3), prior=0.5)
        self.assertEqual((v.answer, v.reads, v.yes, v.no), (UNSURE, 6, 3, 3))

    def test_failed_reads_use_budget_and_carry_no_evidence(self):
        v = Sequential(0.02, 0.06, max_reads=4).decide(
            reader([ReadError(), ReadError(), ReadError(), True]), prior=0.5)
        self.assertEqual((v.answer, v.reads, v.yes), (UNSURE, 4, 1))

    def test_one_disagreeing_read_is_outvoted(self):
        v = Sequential(0.02, 0.06).decide(reader([False, True, True, True, True]), prior=0.98)
        self.assertEqual(v.answer, PRESENT)
        self.assertEqual(v.no, 1)

    def test_bad_parameters_are_rejected(self):
        for kw in (dict(e_fp=0.0, e_fn=0.1), dict(e_fp=0.1, e_fn=0.5),
                   dict(e_fp=0.1, e_fn=0.1, alpha=0.0), dict(e_fp=0.1, e_fn=0.1, beta=0.7),
                   dict(e_fp=0.1, e_fn=0.1, max_reads=0), dict(e_fp=0.1, e_fn=0.1, repeat=1.0)):
            with self.assertRaises(ValueError):
                Sequential(**kw)

    def test_error_rates_stay_under_the_targets_when_calibrated(self):
        # 40000 seeded trials at an even prior. Among "present" answers the
        # share that were really absent must be <= alpha, and the same for
        # "absent" answers and beta.
        rng = random.Random(5)
        s = Sequential(0.05, 0.10, alpha=0.01, beta=0.01, max_reads=12)
        said = {PRESENT: [0, 0], ABSENT: [0, 0], UNSURE: [0, 0]}
        for _ in range(40000):
            truth = rng.random() < 0.5
            v = s.decide(noisy(rng, truth, 0.05, 0.10), prior=0.5)
            said[v.answer][truth] += 1
        wrong_present = said[PRESENT][False] / sum(said[PRESENT])
        wrong_absent = said[ABSENT][True] / sum(said[ABSENT])
        self.assertLessEqual(wrong_present, 0.01)
        self.assertLessEqual(wrong_absent, 0.01)
        self.assertLess(sum(said[UNSURE]), 400)

    def test_repeat_model_discounts_a_second_identical_read(self):
        plain = Sequential(0.02, 0.06, alpha=1e-3)
        aware = Sequential(0.02, 0.06, alpha=1e-3, repeat=0.5, max_reads=10)
        self.assertEqual(plain.decide(reader([True, True]), 0.5).reads, 2)
        v = aware.decide(reader([True] * 10), 0.5)
        self.assertEqual(v.answer, PRESENT)
        self.assertGreater(v.reads, 2)

    def test_observe_needs_a_lead_and_gives_up_on_a_tie(self):
        s = Sequential(0.02, 0.06, max_reads=8)
        self.assertEqual(s.lead, 3)
        self.assertEqual(s.observe(reader([1, 0, 0, 0, 0])), (True, 0, 5))
        ok, _, used = s.observe(reader([0, 1] * 4))
        self.assertEqual((ok, used), (False, 8))


class TestOtherVerifiers(unittest.TestCase):
    def test_single_believes_the_first_good_read(self):
        v = Single().decide(reader([ReadError(), False]))
        self.assertEqual((v.answer, v.reads), (ABSENT, 2))
        self.assertEqual(Single().decide(reader([ReadError()] * 3)).answer, UNSURE)

    def test_majority_stops_early_and_handles_ties(self):
        self.assertEqual(MajorityOf(3).decide(reader([True, True])).reads, 2)
        self.assertEqual(MajorityOf(3).decide(reader([True, False, False])).answer, ABSENT)
        v = MajorityOf(3, max_reads=3).decide(reader([True, ReadError(), False]))
        self.assertEqual(v.answer, UNSURE)
        with self.assertRaises(ValueError):
            MajorityOf(4)

    def test_estimate_rates(self):
        data = [(True, True)] * 90 + [(True, False)] * 10 + [(False, False)] * 95 + [(False, True)] * 5
        e_fp, e_fn = estimate_rates(data)
        self.assertAlmostEqual(e_fp, 6 / 102)
        self.assertAlmostEqual(e_fn, 11 / 102)
        # a clean sample must not claim a perfect reader
        self.assertEqual(estimate_rates([(True, True), (False, False)] * 500), (0.005, 0.005))
        with self.assertRaises(ValueError):
            estimate_rates([(True, True)] * 10)

    def test_estimate_repeat(self):
        pairs = [(True, False, False)] * 6 + [(True, False, True)] * 4 + [(True, True, False)] * 50
        self.assertAlmostEqual(estimate_repeat(pairs), 7 / 12)
        with self.assertRaises(ValueError):
            estimate_repeat([(True, True, True)])


class TestFaultModel(unittest.TestCase):
    def test_misread_rates_match_the_settings(self):
        pms = FakePMS(Faults(seed=3, fp=0.05, fn=0.10))
        pms.tickets.append({"room": 1, "text": "towels (ref aaaa)"})
        n = 20000
        fn = sum(not pms.has_tag("tickets", "text", "ref aaaa") for _ in range(n)) / n
        fp = sum(pms.has_tag("tickets", "text", "ref zzzz") for _ in range(n)) / n
        self.assertAlmostEqual(fn, 0.10, delta=0.01)
        self.assertAlmostEqual(fp, 0.05, delta=0.01)

    def test_no_misreads_means_no_extra_random_draws(self):
        a, b = Faults(seed=1), Faults(seed=1)
        self.assertFalse(a.misread("k", True))
        self.assertEqual(a.rng.random(), b.rng.random())

    def test_burst_makes_wrong_reads_repeat(self):
        f = Faults(seed=2, fp=0.05, fn=0.05, burst=0.8)
        again = wrong = 0
        prev = False
        for _ in range(50000):
            w = f.misread("k", True)
            if prev:
                wrong += 1
                again += w
            prev = w
        self.assertAlmostEqual(again / wrong, 0.8, delta=0.03)

    def test_count_and_field_reads_can_be_wrong_too(self):
        pms = FakePMS(Faults(seed=4, fp=0.2, fn=0.2))
        pms.sms.append({"res": "R1", "text": "hi"})
        pms.fields[("R1", "room")] = 5
        counts = {pms.count("sms", {"res": "R1", "text": "hi"}) for _ in range(300)}
        self.assertEqual(counts, {0, 1, 2})
        self.assertEqual({pms.field_is("R1", "room", 5) for _ in range(300)}, {True, False})


class RunnerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "j.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def step(self, world, answers, act=None):
        """A step on a one-item world whose reads are scripted."""
        read = reader(answers)
        return Step("charge", act or (lambda: world.append("charge")),
                    lambda: None, lambda before: read())


class TestRunnerWithVerifier(RunnerCase):
    def test_one_wrong_absent_read_does_not_cause_a_second_click(self):
        world = []
        j = Journal(self.path)
        seq = Runner(j, verifier=Sequential(0.02, 0.06))
        self.assertEqual(seq.run("t", [self.step(world, [False, True, True, True, True])]), DONE)
        self.assertEqual(world, ["charge"])
        rec = j.records[-1]
        self.assertEqual((rec["k"], rec["no"]), ("done", 1))
        self.assertGreater(rec["p"], 0.999)
        j.close()

    def test_default_runner_double_clicks_on_the_same_wrong_read(self):
        world = []
        j = Journal(self.path)
        self.assertEqual(Runner(j).run("t", [self.step(world, [False, True])]), DONE)
        self.assertEqual(world, ["charge", "charge"])
        j.close()

    def test_one_wrong_present_read_does_not_hide_a_failed_click(self):
        world = []
        calls = []

        def act():
            calls.append(1)
            if len(calls) == 1:
                raise ActionError("element not found")      # nothing applied
            world.append("charge")
        # after the error: a false "present", then the truth
        answers = [True, False, False, False, False, True, True, True]
        j = Journal(self.path)
        st = Runner(j, verifier=Sequential(0.02, 0.06)).run("t", [self.step(world, answers, act)])
        self.assertEqual((st, world), (DONE, ["charge"]))
        j.close()
        world2, calls2 = [], []

        def act2():
            calls2.append(1)
            raise ActionError("element not found")
        j2 = Journal(self.path + "2")
        st = Runner(j2).run("t", [self.step(world2, [True], act2)])
        self.assertEqual((st, world2), (DONE, []))            # recorded as done, never happened
        j2.close()

    def test_unresolvable_reads_escalate_with_the_evidence(self):
        world = []
        j = Journal(self.path)
        r = Runner(j, verifier=Sequential(0.05, 0.05, max_reads=6))
        self.assertEqual(r.run("t", [self.step(world, [True, False] * 3)]), ESCALATED)
        rec = j.records[-1]
        self.assertEqual((rec["k"], rec["why"], rec["yes"], rec["no"]),
                         ("escalated", "cannot verify after act", 3, 3))
        self.assertEqual(world, ["charge"])
        self.assertEqual(r.run("t", [self.step(world, [True] * 9)]), ESCALATED)   # stays stopped
        self.assertEqual(world, ["charge"])
        j.close()

    def test_restart_uses_the_even_prior(self):
        world = ["charge"]
        j = Journal(self.path)
        j.append({"t": "t", "i": 0, "k": "intent", "n": "charge", "before": None})
        st = Runner(j, verifier=Sequential(0.02, 0.06)).run("t", [self.step(world, [True, True, True])])
        self.assertEqual((st, world), (DONE, ["charge"]))
        self.assertEqual((j.records[-1]["how"], j.records[-1]["reads"]), ("found", 2))
        j.close()

    def test_wrong_observation_is_outvoted_before_acting(self):
        # count mode: one stray "1" among the reads of an empty table must not
        # become the baseline, or a sent text would never look sent.
        pms = FakePMS()
        row = {"res": "R1", "text": "hello"}
        answers = iter([1, 0, 0, 0, 0])
        step = Step("sms+", lambda: pms.append("sms", row, ["t", 0]),
                    lambda: next(answers), lambda before: pms.count("sms", row) > before)
        j = Journal(self.path)
        self.assertEqual(Runner(j, verifier=Sequential(0.02, 0.06)).run("t", [step]), DONE)
        self.assertEqual(j.records[0]["before"], 0)
        self.assertEqual(len(pms.sms), 1)
        j.close()

    def _many(self, make, n, rates, seed):
        dup = missing = esc = 0
        faults = Faults(seed=seed, **rates)
        for t in range(n):
            pms = FakePMS(faults)
            path = os.path.join(self.tmp.name, "m%d.jsonl" % t)
            status = None
            for _ in range(50):
                j = Journal(path, fsync=False)
                try:
                    status = make(j).run("t%d" % t, build(pms, "t%d" % t, "late_checkout", "R1", 204, "tag"))
                    break
                except Crash:
                    pass
                finally:
                    j.close()
            os.remove(path)
            d, m = audit(pms, "t%d" % t, "late_checkout", "R1", 204)
            dup += d
            esc += status == ESCALATED
            missing += m if status == DONE else 0
        return dup, missing, esc

    def test_sequential_beats_single_read_under_misreads_and_crashes(self):
        rates = dict(fail_before=0.05, ack_lost=0.05, noop=0.02, crash=0.03,
                     read_fail=0.03, fp=0.03, fn=0.07)
        single = self._many(lambda j: Runner(j), 600, rates, 11)
        seq = self._many(lambda j: Runner(j, verifier=Sequential(0.03, 0.07, 1e-4, 1e-4, 12)),
                         600, rates, 11)
        self.assertGreater(single[0], 50)                 # many duplicate side effects
        self.assertLessEqual(seq[0] + seq[1], 2)
        self.assertLess(seq[2], 30)


class TestReport(RunnerCase):
    def test_summary_counts_steps_escalations_and_open_intents(self):
        j = Journal(self.path)
        r = Runner(j, verifier=Sequential(0.05, 0.05, max_reads=6))
        r.run("a", [self.step([], [False, True, True, True, True])])
        r.run("b", [self.step([], [True, False] * 3)])
        j.append({"t": "c", "i": 0, "k": "intent", "n": "charge", "before": None})
        j.close()
        s = report.summarise(load(self.path))
        c = s["steps"]["charge"]
        self.assertEqual((s["tasks"], c["done"], c["escalated"], c["contested"]), (3, 1, 1, 1))
        self.assertEqual(s["escalations"], {"cannot verify after act": 1})
        self.assertEqual(s["open_intents"], ["c step 0 (charge)"])
        self.assertEqual(s["weakest"][0][1:4], ("a", 0, "charge"))
        text = report.render(s)
        self.assertIn("cannot verify after act x1", text)

    def test_report_reads_a_journal_without_evidence_fields(self):
        j = Journal(self.path)
        Runner(j).run("a", [self.step([], [True])])
        j.close()
        s = report.summarise(load(self.path))
        self.assertEqual(s["steps"]["charge"]["checks"], 0)
        self.assertIn("n/a", report.render(s))

    def test_cli(self):
        self.assertEqual(report.main(["report.py"]), 2)


if __name__ == "__main__":
    unittest.main()
