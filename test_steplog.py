import os
import subprocess
import sys
import tempfile
import unittest

from steplog import (Journal, Runner, DoneOnlyRunner, NaiveRunner, Step, ActionError,
                     ReadError, JournalCorrupt, DONE, ESCALATED)
from fakepms import FakePMS, Faults, Crash, build, audit, tag_for

HERE = os.path.dirname(os.path.abspath(__file__))


class Script:
    """Faults in a fixed order instead of random ones."""

    def __init__(self, actions=(), reads=()):
        self.a, self.r = list(actions), list(reads)

    def action(self):
        return self.a.pop(0) if self.a else "ok"

    def read(self):
        return self.r.pop(0) if self.r else False


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "j.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def go(self, pms, cls=Runner, mode="count", template="late_checkout"):
        """Run to completion, restarting after every crash. -> (status, restarts)"""
        for restarts in range(20):
            j = Journal(self.path)
            try:
                return cls(j).run("t1", build(pms, "t1", template, "R1", 204, mode)), restarts
            except Crash:
                pass
            finally:
                j.close()
        self.fail("too many crashes")


class TestRunner(Base):
    def test_clean_run(self):
        pms = FakePMS()
        self.assertEqual(self.go(pms), (DONE, 0))
        self.assertEqual(audit(pms, "t1", "late_checkout", "R1", 204), (0, 0))
        self.assertEqual(pms.acts, 4)

    def test_timeout_after_apply_is_not_repeated(self):
        pms = FakePMS(Script(["ack_lost"]))
        self.assertEqual(self.go(pms)[0], DONE)
        self.assertEqual(len(pms.charges), 1)
        self.assertEqual(pms.acts, 4)

    def test_baselines_double_charge_on_timeout(self):
        for cls in (NaiveRunner, DoneOnlyRunner):
            if os.path.exists(self.path):
                os.remove(self.path)
            pms = FakePMS(Script(["ack_lost"]))
            self.go(pms, cls)
            self.assertEqual(len(pms.charges), 2, cls.__name__)

    def test_failure_before_apply_is_retried(self):
        pms = FakePMS(Script(["fail_before", "fail_before"]))
        self.assertEqual(self.go(pms)[0], DONE)
        self.assertEqual(audit(pms, "t1", "late_checkout", "R1", 204), (0, 0))

    def test_silent_noop_is_caught(self):
        pms = FakePMS(Script(["noop"]))
        self.assertEqual(self.go(pms)[0], DONE)
        self.assertEqual(len(pms.charges), 1)
        pms2 = FakePMS(Script(["noop"]))
        os.remove(self.path)
        self.go(pms2, DoneOnlyRunner)
        self.assertEqual(len(pms2.charges), 0)

    def test_crash_after_apply_resumes_without_repeat(self):
        for mode in ("count", "tag"):
            if os.path.exists(self.path):
                os.remove(self.path)
            pms = FakePMS(Script(["crash_after", "ok", "crash_after", "crash_after"]))
            status, restarts = self.go(pms, mode=mode)
            self.assertEqual((status, restarts), (DONE, 3))
            self.assertEqual(audit(pms, "t1", "late_checkout", "R1", 204), (0, 0))
            self.assertEqual(pms.acts, 4)

    def test_crash_before_apply_redoes_the_step(self):
        pms = FakePMS(Script(["crash_before"]))
        self.assertEqual(self.go(pms), (DONE, 1))
        self.assertEqual(len(pms.charges), 1)

    def test_naive_restart_repeats_everything(self):
        pms = FakePMS(Script(["ok", "ok", "ok", "crash_after"]))
        self.go(pms, NaiveRunner)
        self.assertEqual(len(pms.charges), 2)
        self.assertEqual(len(pms.sms), 2)

    def test_existing_identical_row_is_not_mistaken_for_ours(self):
        pms = FakePMS(Script(["crash_before"]))
        pms.charges.append({"res": "R1", "amount": 40, "desc": "Late checkout fee", "_src": "old"})
        self.assertEqual(self.go(pms)[0], DONE)
        self.assertEqual(len(pms.charges), 2)

    def test_unreadable_screen_escalates_and_stays_escalated(self):
        pms = FakePMS(Script(["ack_lost"], [False, True, True, True]))
        self.assertEqual(self.go(pms)[0], ESCALATED)
        self.assertEqual(len(pms.charges), 1)
        acts = pms.acts
        self.assertEqual(self.go(pms)[0], ESCALATED)
        self.assertEqual(pms.acts, acts)

    def test_attempts_exhausted_escalates(self):
        pms = FakePMS(Script(["fail_before"] * 4))
        self.assertEqual(self.go(pms)[0], ESCALATED)
        self.assertEqual(len(pms.charges), 0)

    def test_count_mode_is_fooled_by_a_clerk_and_tag_mode_is_not(self):
        row = {"room": 204, "text": "Extra towels"}
        # ops: 1 observe, 2 act (does nothing), clerk writes at op 3, 3 verify
        pms = FakePMS(Script(["noop"]))
        pms.scheduled.append((3, "tickets", row))
        self.assertEqual(self.go(pms, template="service_request")[0], DONE)
        self.assertEqual(audit(pms, "t1", "service_request", "R1", 204), (0, 1))
        os.remove(self.path)
        pms = FakePMS(Script(["noop"]))
        pms.scheduled.append((2, "tickets", row))
        self.assertEqual(self.go(pms, mode="tag", template="service_request")[0], DONE)
        self.assertEqual(audit(pms, "t1", "service_request", "R1", 204), (0, 0))
        self.assertTrue(any(tag_for("t1", 0) in t["text"] for t in pms.tickets))

    def test_random_faults_never_duplicate(self):
        dup = 0
        for seed in range(300):
            if os.path.exists(self.path):
                os.remove(self.path)
            pms = FakePMS(Faults(seed, fail_before=.1, ack_lost=.1, noop=.05, crash=.1, read_fail=.05))
            self.go(pms, template="room_move")
            dup += audit(pms, "t1", "room_move", "R1", 204)[0]
        self.assertEqual(dup, 0)


class TestJournal(Base):
    def test_torn_tail_is_dropped_and_file_stays_appendable(self):
        j = Journal(self.path)
        j.append({"t": "a", "i": 0, "k": "intent", "before": 1})
        j.close()
        with open(self.path, "ab") as f:
            f.write(b'{"t":"a","i":0,"k":"do')
        j = Journal(self.path)
        self.assertEqual(j.state("a"), (set(), {0: 1}, False))
        j.append({"t": "a", "i": 0, "k": "done"})
        j.close()
        j = Journal(self.path)
        self.assertEqual(j.state("a"), ({0}, {}, False))
        j.close()

    def test_bad_line_in_the_middle_raises(self):
        with open(self.path, "wb") as f:
            f.write(b'{"t":"a","i":0,"k":"intent"}\ngarbage\n{"t":"a","i":0,"k":"done"}\n')
        with self.assertRaises(JournalCorrupt):
            Journal(self.path)

    def test_tasks_do_not_mix(self):
        j = Journal(self.path)
        j.append({"t": "a", "i": 0, "k": "done"})
        j.append({"t": "b", "i": 0, "k": "escalated"})
        self.assertEqual(j.state("a"), ({0}, {}, False))
        self.assertTrue(j.state("b")[2])
        j.close()

    def test_generic_step_without_the_fake(self):
        box = []
        calls = {"n": 0}

        def act():
            calls["n"] += 1
            box.append("x")
            if calls["n"] == 1:
                raise ActionError("timeout")
        j = Journal(self.path)
        s = Step("push", act, lambda: len(box), lambda before: len(box) > before)
        self.assertEqual(Runner(j).run("g", [s]), DONE)
        self.assertEqual(box, ["x"])
        j.close()


class TestRealKill(unittest.TestCase):
    def test_process_killed_after_charge_does_not_charge_twice(self):
        with tempfile.TemporaryDirectory() as d:
            demo = os.path.join(HERE, "demo.py")
            r = subprocess.run([sys.executable, demo, "worker", d, "1"], capture_output=True)
            self.assertEqual(r.returncode, 9)
            r = subprocess.run([sys.executable, demo, "worker", d], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0)
            self.assertIn("actions this run: 3", r.stdout)
            pms = FakePMS.load(os.path.join(d, "pms.json"))
            self.assertEqual((len(pms.charges), len(pms.tickets), len(pms.sms)), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
