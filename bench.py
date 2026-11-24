"""Run the same faulty workload through each runner and count what went wrong.

    python bench.py            # 20000 tasks per row, a few minutes
    python bench.py 2000       # quicker, does not rewrite bench_results.json

Part 1: reads are always right (the steplog 1 benchmark, unchanged).
Part 2: reads can be wrong. A verifier is tuned on one synthetic workload and
        then scored on two held-out workloads it was not tuned on: one with a
        different seed and worse fault rates, and one where wrong reads also
        come in bursts.

All data is synthetic. The fault rates and cost weights are my assumptions.
"""
import json
import os
import random
import sys
import tempfile
import time

from steplog import Journal, Runner, DoneOnlyRunner, NaiveRunner, DONE, ESCALATED
from fakepms import FakePMS, Faults, Crash, TEMPLATES, CLERK_ROWS, build, audit, tag_for
from verify import Single, MajorityOf, Sequential, estimate_rates, estimate_repeat

RATES = dict(fail_before=0.05, ack_lost=0.05, noop=0.02, crash=0.03, read_fail=0.03)
RUNNERS = (
    ("no journal, restart from step 0", NaiveRunner, "count"),
    ("checkpoint after each step", DoneOnlyRunner, "count"),
    ("steplog, verify by count", Runner, "count"),
    ("steplog, verify by reference tag", Runner, "tag"),
)

# Part 2 workloads. TUNE is the only one used to pick parameters.
TUNE = dict(RATES, fp=0.02, fn=0.06)
HELD = dict(fail_before=0.06, ack_lost=0.07, noop=0.03, crash=0.05, read_fail=0.05,
            fp=0.04, fn=0.09)
BURST = dict(HELD, burst=0.5)
TUNE_SEED, HELD_SEED, BURST_SEED = 7, 1011, 2024
CLERK = 0.3

# What one outcome costs, in units of one read (a screenshot plus a model call).
COST = dict(read=1.0, escalated=40.0, missing=400.0, duplicate=1000.0)
GRID = [(a, b, m) for a in (1e-2, 1e-3, 1e-4) for b in (1e-2, 1e-3, 1e-4, 1e-5)
        for m in (6, 10)]


def run_config(label, make, mode, n, clerk, tmp, seed=7, rates=RATES):
    """make(journal) -> runner. Same tasks and clerk events for every row that
    shares a seed."""
    world = random.Random(seed)
    faults = Faults(seed=seed + 1, **rates)
    tot = dict(tasks=n, correct=0, escalated=0, failed=0, dup=0, missing=0,
               tasks_with_dup=0, acts=0, reads=0, restarts=0)
    path = os.path.join(tmp, "j.jsonl")
    for t in range(n):
        task = "t%d" % t
        template = world.choice(TEMPLATES)
        res, room = "R%d" % t, world.randrange(101, 400)
        clerk_roll, clerk_at = world.random(), world.randrange(1, 9)
        pms = FakePMS(faults)
        if clerk_roll < clerk:
            _, table, mk = CLERK_ROWS[template][0]
            pms.scheduled.append((clerk_at, table, mk(res, room)))
        if os.path.exists(path):
            os.remove(path)
        status = None
        for attempt in range(50):
            j = Journal(path, fsync=False)   # a restart reloads the file
            try:
                status = make(j).run(task, build(pms, task, template, res, room, mode))
                break
            except Crash:
                tot["restarts"] += 1
            finally:
                j.close()
        dup, missing = audit(pms, task, template, res, room)
        tot["dup"] += dup
        tot["tasks_with_dup"] += dup > 0
        tot["acts"] += pms.acts
        tot["reads"] += pms.reads
        if status == ESCALATED:
            tot["escalated"] += 1
        elif status != DONE:
            tot["failed"] += 1
        else:
            tot["missing"] += missing
            tot["correct"] += dup == 0 and missing == 0
    tot["label"], tot["clerk"] = label, clerk
    tot["cost_per_task"] = round((COST["read"] * tot["reads"] + COST["escalated"] * tot["escalated"]
                                  + COST["missing"] * tot["missing"]
                                  + COST["duplicate"] * tot["dup"]) / n, 3)
    return tot


def calibrate(rates, seed, n=1000):
    """Stand-in for a person labelling screenshots: n reads of a screen that
    has the row and n of one that does not, with the truth known."""
    pms = FakePMS(Faults(seed=seed, fp=rates["fp"], fn=rates["fn"]))
    labelled = []
    for i in range(n):
        labelled.append((False, pms.has_tag("tickets", "text", tag_for("cal", i))))
    for i in range(n):
        tag = tag_for("cal", i)
        pms.tickets.append({"room": 1, "text": "x (%s)" % tag})
        labelled.append((True, pms.has_tag("tickets", "text", tag)))
    return estimate_rates(labelled)


def calibrate_repeat(rates, seed, n=4000):
    """Two looks at the same screen, truth known, drawn from the burst fault
    model with its own seed."""
    pms = FakePMS(Faults(seed=seed, fp=rates["fp"], fn=rates["fn"], burst=rates.get("burst", 0.0)))
    pairs = []
    for i in range(n):
        tag = tag_for("rep", i)
        if i % 2:
            pms.tickets.append({"room": 1, "text": "x (%s)" % tag})
        read = lambda: pms.has_tag("tickets", "text", tag)
        pairs.append((bool(i % 2), read(), read()))
    return estimate_repeat(pairs)


def journal_speed(tmp, n=2000):
    out = {}
    for fsync in (False, True):
        p = os.path.join(tmp, "speed.jsonl")
        if os.path.exists(p):
            os.remove(p)
        j = Journal(p, fsync=fsync)
        t0 = time.perf_counter()
        for i in range(n):
            j.append({"t": "x", "i": i, "k": "intent", "n": "tickets+", "before": 3})
        out["fsync" if fsync else "no_fsync"] = round((time.perf_counter() - t0) / n * 1e6, 1)
        j.close()
    return out


HEAD1 = ("| runner | exactly right | tasks with a duplicate | duplicates | missing | escalated | gave up | actions/task | reads/task |\n"
         "|---|---|---|---|---|---|---|---|---|")
HEAD2 = ("| verifier | exactly right | duplicates | missing | escalated | reads/task | cost/task |\n"
         "|---|---|---|---|---|---|---|")


def row1(r):
    n = r["tasks"]
    return "| %s | %.2f%% | %.2f%% | %d | %d | %d | %d | %.2f | %.2f |" % (
        r["label"], 100.0 * r["correct"] / n, 100.0 * r["tasks_with_dup"] / n,
        r["dup"], r["missing"], r["escalated"], r["failed"], r["acts"] / n, r["reads"] / n)


def row2(r):
    n = r["tasks"]
    return "| %s | %.2f%% | %d | %d | %d | %.2f | %.2f |" % (
        r["label"], 100.0 * r["correct"] / n, r["dup"], r["missing"], r["escalated"],
        r["reads"] / n, r["cost_per_task"])


def part1(n, tmp):
    rows = []
    for clerk in (0.0, 0.3):
        print("\nreads always right, clerk enters the same ticket by hand in %d%% of tasks, %d tasks per row"
              % (clerk * 100, n))
        print(HEAD1)
        for label, cls, mode in RUNNERS:
            r = run_config(label, cls, mode, n, clerk, tmp)
            rows.append(r)
            print(row1(r))
    return rows


def part2(n, tmp):
    e_fp, e_fn = calibrate(TUNE, seed=99)
    print("\nreads can be wrong. Rates estimated from 2000 labelled synthetic reads: "
          "e_fp=%.4f e_fn=%.4f (true tune rates %.2f, %.2f)" % (e_fp, e_fn, TUNE["fp"], TUNE["fn"]))
    n_tune = max(200, n // 2)
    grid = []
    for a, b, m in GRID:
        mk = lambda j, a=a, b=b, m=m: Runner(j, verifier=Sequential(e_fp, e_fn, a, b, m))
        r = run_config("alpha=%g beta=%g max_reads=%d" % (a, b, m), mk, "tag", n_tune, CLERK,
                       tmp, TUNE_SEED, TUNE)
        grid.append(dict(alpha=a, beta=b, max_reads=m, cost_per_task=r["cost_per_task"],
                         dup=r["dup"], missing=r["missing"], escalated=r["escalated"],
                         reads_per_task=round(r["reads"] / n_tune, 3)))
    best = min(grid, key=lambda g: (g["cost_per_task"], g["alpha"] * g["beta"]))
    print("tuned on %d tune tasks over %d settings: alpha=%g beta=%g max_reads=%d"
          % (n_tune, len(GRID), best["alpha"], best["beta"], best["max_reads"]))
    repeat = calibrate_repeat(BURST, seed=4242)
    print("repeat rate estimated from 4000 labelled pairs of synthetic burst reads: %.3f (true %.2f)"
          % (repeat, BURST["burst"]))
    verifiers = (
        ("one read (steplog 1)", lambda: Single()),
        ("majority of 3", lambda: MajorityOf(3)),
        ("majority of 5", lambda: MajorityOf(5)),
        ("sequential, untuned (alpha=beta=1e-3, 8 reads)", lambda: Sequential(e_fp, e_fn)),
        ("sequential, tuned", lambda: Sequential(e_fp, e_fn, best["alpha"], best["beta"],
                                                 best["max_reads"])),
        ("sequential, tuned, burst-aware", lambda: Sequential(e_fp, e_fn, best["alpha"], best["beta"],
                                                              best["max_reads"], repeat=repeat)),
    )
    sets = (("tune", "tune workload (seed %d, the settings the parameters were picked on)" % TUNE_SEED,
             TUNE, TUNE_SEED),
            ("held_out", "held-out workload (seed %d, higher fault and misread rates)" % HELD_SEED,
             HELD, HELD_SEED),
            ("held_out_burst", "held-out workload with bursts of wrong reads (seed %d, burst=0.5)"
             % BURST_SEED, BURST, BURST_SEED))
    out = {}
    for key, title, rates, seed in sets:
        print("\n%s, %d tasks per row" % (title, n))
        print(HEAD2)
        out[key] = []
        for label, mkv in verifiers:
            r = run_config(label, lambda j, mkv=mkv: Runner(j, verifier=mkv()), "tag", n,
                           CLERK, tmp, seed, rates)
            out[key].append(r)
            print(row2(r))
    return dict(estimated=dict(e_fp=round(e_fp, 5), e_fn=round(e_fn, 5), repeat=round(repeat, 4)), n_tune=n_tune,
                grid=grid, chosen=best, results=out,
                workloads=dict(tune=TUNE, held_out=HELD, held_out_burst=BURST), cost=COST)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        rows = part1(n, tmp)
        noisy = part2(n, tmp)
        speed = journal_speed(tmp)
    print("\njournal append: %.1f us without fsync, %.1f us with fsync" % (speed["no_fsync"], speed["fsync"]))
    print("total time: %.0f s" % (time.time() - t0))
    if len(sys.argv) > 1:
        print("partial run: bench_results.json not rewritten")
        return
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.json"), "w") as f:
        json.dump({"synthetic": True, "tasks_per_row": n, "rates": RATES, "rows": rows,
                   "noisy_reads": noisy, "journal_append_us": speed}, f, indent=1)


if __name__ == "__main__":
    main()
