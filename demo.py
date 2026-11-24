"""Two short demos against the fake property system (synthetic data).

    python demo.py                 # runs both
    python demo.py worker DIR [N]  # internal: run the task, die after action N

1. One late checkout with a real process kill right after the charge posts.
2. 300 late checkouts where reads are sometimes wrong (4% see a row that is
   not there, 9% miss one that is), once believing a single read and once
   with the sequential verifier, then the journal report for the second run.
"""
import os
import subprocess
import sys
import tempfile

from steplog import Journal, Runner, DONE, ESCALATED
from fakepms import FakePMS, Faults, Crash, build, audit
from verify import Sequential
import report

TASK = ("demo-1", "late_checkout", "R1", 204)


def worker(d, die_after):
    state = os.path.join(d, "pms.json")
    pms = FakePMS.load(state) if os.path.exists(state) else FakePMS()
    real_act = pms._act

    def act(apply):
        real_act(apply)
        pms.save(state)                 # the hotel system now has the change
        if pms.acts == die_after:
            os._exit(9)                 # dies before the done record is written
    pms._act = act
    j = Journal(os.path.join(d, "journal.jsonl"))
    status = Runner(j).run(TASK[0], build(pms, *TASK))
    print("worker finished: %s, actions this run: %d, reads this run: %d" % (status, pms.acts, pms.reads))


NOISY = dict(fail_before=0.05, ack_lost=0.05, noop=0.02, crash=0.03, read_fail=0.03,
             fp=0.04, fn=0.09)


def noisy_run(d, name, make, n=300):
    """n late checkouts through a fake with wrong reads. One journal for all."""
    faults = Faults(seed=21, **NOISY)
    path = os.path.join(d, name + ".jsonl")
    dup = missing = esc = reads = 0
    for t in range(n):
        task, pms = "lc-%d" % t, FakePMS(faults)
        status = None
        for _ in range(50):                 # restart after every simulated crash
            j = Journal(path, fsync=False)
            try:
                status = make(j).run(task, build(pms, task, "late_checkout", "R%d" % t, 204, "tag"))
                break
            except Crash:
                pass
            finally:
                j.close()
        d_, m_ = audit(pms, task, "late_checkout", "R%d" % t, 204)
        dup += d_
        esc += status == ESCALATED
        missing += m_ if status == DONE else 0
        reads += pms.reads
    print("%-22s duplicates: %3d   recorded done but missing: %3d   escalated: %3d   reads/task: %.1f"
          % (name, dup, missing, esc, reads / n))
    return path


def main():
    kill_demo()
    print()
    print("300 late checkouts, reads wrong 4%/9% of the time (synthetic):")
    with tempfile.TemporaryDirectory() as d:
        noisy_run(d, "believe one read", lambda j: Runner(j))
        path = noisy_run(d, "sequential verifier", lambda j: Runner(j, verifier=Sequential(0.04, 0.09)))
        print()
        print("python report.py on the sequential verifier's journal:")
        j = Journal(path)
        j.close()
        print(report.render(report.summarise(j.records)))


def kill_demo():
    with tempfile.TemporaryDirectory() as d:
        me = os.path.abspath(__file__)
        r1 = subprocess.run([sys.executable, me, "worker", d, "1"])
        print("first run exit code: %d (killed right after posting the $40 charge)" % r1.returncode)
        subprocess.run([sys.executable, me, "worker", d], check=True)
        pms = FakePMS.load(os.path.join(d, "pms.json"))
        print("charges on the folio: %d" % len(pms.charges))
        print("tickets: %d, texts sent: %d, departure: %s" % (
            len(pms.tickets), len(pms.sms), pms.fields[("R1", "departure_time")]))
        print("journal:")
        for line in open(os.path.join(d, "journal.jsonl")):
            print("  " + line.rstrip())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else -1)
    else:
        main()
