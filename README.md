# steplog

Run a multi-step task through a UI without doing any step twice, including when the reads of the screen are sometimes wrong.

This is an independent prototype. It is not affiliated with, reviewed by or tested against Lance or any hotel software. All data is synthetic and comes from a fake property system in this repo.

## The problem

An agent that operates hotel software by clicking through it has no idempotency key to send. If it posts a $40 late checkout fee and the confirmation screen times out, or the worker dies right after the click, retrying charges the guest twice and not retrying may leave the fee unposted. The only way to know is to look at the screen.

The first version of steplog did that: write an intent record, act, read the screen, write a done record, and read before any retry. It had one assumption I listed as a limit and did not test: that a read which succeeds is correct. When the reader is a vision model looking at a screenshot, that is not true. A read that wrongly says "present" records a step that never happened. A read that wrongly says "absent" causes a second click, which is the double charge the journal was built to prevent.

When I added wrong reads to the simulator, the first version fell apart. With 2% false "present" and 6% false "absent" reads it left 3822 duplicate side effects across 20,000 tasks. This version is about fixing that.

## Why this matters to Lance

What I could verify from public pages:

- Lance's YC page says its receptionist agent uses vision-based computer-use agents to operate older systems such as on-prem PMS or CRS software when a workflow needs it (https://www.ycombinator.com/companies/lance).
- The software engineering intern posting lists computer-use agent infrastructure, backend systems for multi-step task execution, and reliability, latency and observability of the agent platform (https://jobs.svangel.com/companies/lance-yc-w26-2/jobs/87366517-software-engineering-intern).
- The same YC page describes agents that take requests such as towels and late checkout, route them, and escalate when needed, across call, SMS and email.

What I am assuming: that a vision-based agent can misread a screen, that actions through a UI can time out after they applied, and that escalating to a person is an acceptable outcome. I do not know how Lance's agents verify their actions, what systems they run on or what their error rates are. Nothing here describes their architecture.

## What it does

- `steplog.py`: the journaled runner. A step is three functions, `observe()`, `act()` and `present(before)`. The runner writes an intent before acting and a done record after, and after any error or restart it checks the screen before it acts again. It escalates when it cannot tell.
- `verify.py` (new): decides "is my change on the screen?" from reads that can be wrong. `Sequential` keeps reading until the probability that the change is there is above `1 - alpha` or below `beta`, and answers "unsure" when the read budget runs out. "Unsure" becomes an escalation. `Single` (the old behaviour) and `MajorityOf(k)` are baselines.
- `fakepms.py`: a fake property system with charges, tickets, texts and reservation fields. Actions can fail before applying, apply and then time out, silently do nothing, or crash the worker. Reads can fail, and now they can also succeed with the wrong answer, optionally in bursts where a wrong read tends to repeat.
- `report.py` (new): reads a journal and prints, per step type, how many were done, found after a failure, escalated and contested, plus the done records accepted with the least certainty.
- `bench.py`: runs every runner and verifier over the same synthetic workloads and counts duplicates and missing side effects exactly, using a hidden field on each row that the agent never sees.

## Architecture

```
task steps --> Runner (steplog.py) --> Journal (append-only JSONL, fsync)
                  |   intent -> act -> check -> done / escalated
                  v
              verifier (verify.py)  <-- prior: 0.98 after an ack, 0.5 after an error or restart
                  |   repeated reads until sure, or "unsure"
                  v
              screen reads (fakepms.py in this repo; a model reading a screenshot in real use)

journal --> report.py (per-step counts, escalation reasons, weakest evidence)
```

The verifier is the only new moving part in the runner. With no verifier the runner behaves exactly as before, and the 18 original tests pass unchanged.

## Key decisions and tradeoffs

**A sequential test instead of a fixed number of reads.** After an action that reported success, the change is almost certainly there, so one agreeing read is enough. After a timeout it is a coin flip, so it takes more. A fixed majority vote spends the same reads on both. On the held-out workload below, majority of 5 used 17.76 reads per task and left 325 duplicates. The sequential verifier used 10.36 reads and left 15.

**Two separate error targets.** `alpha` limits steps recorded as done that did not happen. `beta` limits repeated clicks. They are separate because a missed towel ticket and a double charge do not cost the same.

**The prior cannot outvote the screen.** At least one successful read is always required, and the prior is clamped inside the stopping bounds, so a single contrary read always reopens the question. My first implementation let a prior of 1.0 answer "present" when the only read said "no". A test caught it.

**The observation before acting needs a stronger rule than the check after.** In count mode the runner records how many identical rows exist before it acts. My first version accepted that number once two reads agreed. With 4% false positives that produced a wrong baseline often enough that text messages were sent twice, because every later check compared against the wrong number. Tracing the duplicates in the journal showed all but a few were this case. The observation now needs the leading answer to be ahead by a margin derived from the error rates (3 reads at the default settings).

**Escalate instead of guess.** When reads conflict until the budget is spent, the runner writes an escalated record with the read counts and stops. A rerun does nothing until a person clears it.

**What it costs.** Reads. On the tune workload the sequential verifier does 8.97 reads per task where believing one read does 5.40. Each read would be a screenshot and a model call in real use.

## Install and usage

Python 3, standard library only. Nothing to install. Tested on Python 3.13.

```
python demo.py                  # 10 to 20 seconds
python -m unittest              # 46 tests, about 30 seconds
python bench.py 2000            # quick pass, 1 to 2 minutes, does not rewrite bench_results.json
python bench.py                 # 20000 tasks per row; took 11 minutes here, mostly file I/O
python report.py journal.jsonl  # summarise any steplog journal
```

```python
from steplog import Journal, Runner, Step
from verify import Sequential, estimate_rates

e_fp, e_fn = estimate_rates(labelled)        # (truth, read) pairs a person checked
step = Step("post late fee",
            act=lambda: ui.post_charge(res, 40, "Late checkout fee (ref 3fa1c2d4)"),
            observe=lambda: None,
            present=lambda before: ui.sees_text("ref 3fa1c2d4"))
runner = Runner(Journal("tasks.jsonl"), verifier=Sequential(e_fp, e_fn, alpha=1e-3, beta=1e-3))
status = runner.run("task-812", [step])      # "done" or "needs_human"
```

## A real example

Output of `python demo.py`. The first part kills a worker process with `os._exit` right after the charge is posted and resumes it. The second part runs 300 late checkouts with wrong reads.

```
first run exit code: 9 (killed right after posting the $40 charge)
worker finished: done, actions this run: 3, reads this run: 6
charges on the folio: 1
tickets: 1, texts sent: 1, departure: 14:00
journal:
  {"t":"demo-1","i":0,"k":"intent","n":"charges+","before":0}
  {"t":"demo-1","i":0,"k":"done","how":"found"}
  {"t":"demo-1","i":1,"k":"intent","n":"set departure_time","before":null}
  {"t":"demo-1","i":1,"k":"done","how":"acted"}
  {"t":"demo-1","i":2,"k":"intent","n":"tickets+","before":0}
  {"t":"demo-1","i":2,"k":"done","how":"acted"}
  {"t":"demo-1","i":3,"k":"intent","n":"sms+","before":0}
  {"t":"demo-1","i":3,"k":"done","how":"acted"}

300 late checkouts, reads wrong 4%/9% of the time (synthetic):
believe one read       duplicates:  95   recorded done but missing:   4   escalated:   0   reads/task: 6.0
sequential verifier    duplicates:   0   recorded done but missing:   0   escalated:   1   reads/task: 10.5

python report.py on the sequential verifier's journal:
tasks: 300
step                  done  acted  found  escalated reads/final check  contested
charges+               299    277     22          1              1.41         31
set departure_time     299    275     24          0              1.43         31
sms+                   299    281     18          0              1.45         34
tickets+               299    283     16          0              1.34         24
escalations: cannot verify after act x1
open intents (worker died mid-step, next run will look before acting): none
done records accepted with the least certainty:
  lc-11 step 2 tickets+ p=0.999015  reads said yes 4, no 4
  lc-106 step 1 set departure_time p=0.999095  reads said yes 3, no 1
  lc-130 step 3 sms+ p=0.999095  reads said yes 3, no 1
  lc-144 step 0 charges+ p=0.999095  reads said yes 3, no 1
  lc-167 step 1 set departure_time p=0.999095  reads said yes 3, no 1
```

In the first part the resumed run finds the charge (`"how":"found"`) and does 3 actions instead of 4. In the second, believing one read left 95 duplicate side effects in 300 tasks and the sequential verifier left none and escalated 1 task. The report shows task lc-11 step 2 was accepted after four reads said yes and four said no, which is the kind of record I would want a person to be able to find.

## Tests and verified results

`python -m unittest` runs 46 tests: the 18 original ones, and 28 new ones covering the verifiers (including a 40,000 trial check that wrong answers stay under `alpha` and `beta` when the rates are right), the misread fault model, the runner with a verifier (wrong "absent" read, wrong "present" read, conflicting reads, wrong observation, restart), and the report. They also pass with `python -W error -m unittest`.

All numbers below are from one run of `python bench.py` on synthetic data, saved in `bench_results.json`. The fault rates and cost weights are my assumptions, not measurements of any real system.

### Part 1: reads are always right (unchanged from the first version)

Each action has a 5% chance of failing before it applies, 5% of applying and then timing out, 2% of silently doing nothing, and 3% of crashing the worker. Each read fails 3% of the time. 20,000 tasks per row.

No one else touching the system:

| runner | exactly right | tasks with a duplicate | duplicates | missing | escalated | gave up | actions/task | reads/task |
|---|---|---|---|---|---|---|---|---|
| no journal, restart from step 0 | 71.31% | 22.34% | 7067 | 1554 | 0 | 13 | 4.44 | 0.00 |
| checkpoint after each step | 74.13% | 19.29% | 4495 | 1618 | 0 | 12 | 4.20 | 0.00 |
| steplog, verify by count | 99.97% | 0.00% | 0 | 0 | 5 | 0 | 3.99 | 7.20 |
| steplog, verify by reference tag | 99.99% | 0.00% | 0 | 0 | 2 | 0 | 3.99 | 5.15 |

A front desk clerk enters the same housekeeping ticket by hand in 30% of tasks:

| runner | exactly right | tasks with a duplicate | duplicates | missing | escalated | gave up | actions/task | reads/task |
|---|---|---|---|---|---|---|---|---|
| no journal, restart from step 0 | 71.31% | 22.34% | 7067 | 1554 | 0 | 13 | 4.44 | 0.00 |
| checkpoint after each step | 74.13% | 19.29% | 4495 | 1618 | 0 | 12 | 4.20 | 0.00 |
| steplog, verify by count | 99.38% | 0.00% | 0 | 119 | 5 | 0 | 3.99 | 7.19 |
| steplog, verify by reference tag | 99.99% | 0.00% | 0 | 0 | 2 | 0 | 3.99 | 5.15 |

Checkpointing alone still left a duplicate in 19.29% of tasks, because the dangerous moment is inside a step. Verifying by count is fooled by the clerk (119 missing). A reference tag in a free-text field is not.

### Part 2: reads can be wrong

Tag mode for charges and tickets, count mode for guest texts, clerk in 30% of tasks. Misread rates were estimated from 2000 labelled synthetic reads (e_fp=0.0210, e_fn=0.0589). `alpha`, `beta` and the read budget were picked from 24 settings on 10,000 tune tasks by lowest cost, where a read costs 1, an escalation 40, a missing side effect 400 and a duplicate 1000. The pick was alpha=0.001, beta=0.001, 10 reads. Nothing was tuned on the two held-out workloads.

Tune workload (seed 7; fp 2%, fn 6%):

| verifier | exactly right | duplicates | missing | escalated | reads/task | cost/task |
|---|---|---|---|---|---|---|
| one read (steplog 1) | 82.30% | 3822 | 121 | 10 | 5.40 | 198.94 |
| majority of 3 | 97.30% | 540 | 6 | 4 | 10.89 | 38.02 |
| majority of 5 | 99.42% | 114 | 0 | 2 | 16.32 | 22.02 |
| sequential, untuned (alpha=beta=1e-3, 8 reads) | 99.67% | 3 | 33 | 30 | 8.97 | 9.84 |
| sequential, tuned | 99.86% | 2 | 26 | 1 | 8.97 | 9.59 |
| sequential, tuned, burst-aware | 87.50% | 0 | 25 | 2474 | 11.64 | 17.09 |

Held-out workload (seed 1011; fp 4%, fn 9%, and higher action fault rates than the verifier was calibrated for):

| verifier | exactly right | duplicates | missing | escalated | reads/task | cost/task |
|---|---|---|---|---|---|---|
| one read (steplog 1) | 73.36% | 5783 | 410 | 32 | 5.80 | 303.21 |
| majority of 3 | 93.41% | 1283 | 39 | 45 | 11.85 | 76.87 |
| majority of 5 | 98.31% | 325 | 6 | 10 | 17.76 | 34.15 |
| sequential, untuned (alpha=beta=1e-3, 8 reads) | 98.38% | 13 | 118 | 194 | 10.32 | 13.71 |
| sequential, tuned | 99.10% | 15 | 135 | 31 | 10.36 | 13.88 |
| sequential, tuned, burst-aware | 76.92% | 0 | 79 | 4537 | 13.43 | 24.08 |

Held-out workload with bursts (seed 2024; same as above, and a wrong read is wrong again on the next look half the time):

| verifier | exactly right | duplicates | missing | escalated | reads/task | cost/task |
|---|---|---|---|---|---|---|
| one read (steplog 1) | 73.33% | 7580 | 372 | 616 | 5.83 | 393.50 |
| majority of 3 | 73.63% | 6475 | 383 | 217 | 12.44 | 344.28 |
| majority of 5 | 78.44% | 4820 | 289 | 182 | 19.01 | 266.15 |
| sequential, untuned (alpha=beta=1e-3, 8 reads) | 88.67% | 1031 | 414 | 939 | 10.66 | 72.37 |
| sequential, tuned | 91.59% | 1050 | 416 | 331 | 10.92 | 72.40 |
| sequential, tuned, burst-aware | 65.23% | 23 | 72 | 6862 | 13.00 | 29.31 |

What I take from this:

- On the held-out workload, believing one read left 5783 duplicates and 410 missing side effects in 20,000 tasks. The sequential verifier left 15 and 135 and escalated 31 tasks, at 10.36 reads per task against 5.80.
- Tuning did not help much. The tuned setting differs from my default only in the read budget. On the held-out workload its cost was 13.88 against 13.71 untuned. It did cut escalations from 194 to 31.
- The 135 missing side effects on the held-out workload are more than I expected from alpha=0.001. The likely reason is that the verifier was calibrated at about half the real false positive rate, but I did not isolate that. The bound is only as good as the calibration.
- Bursts break it. When wrong reads repeat, rereading the same screen is weak evidence, and the sequential verifier left 1050 duplicates. A burst-aware variant that discounts repeated identical reads cut that to 23, but only by escalating 6862 of 20,000 tasks, and it escalates 2474 tasks even on the workload with no bursts. Its repeat rate was estimated from labelled reads drawn from the burst fault model, so that row is not fully held out. I do not consider it usable as is.

## Limitations and next steps

- Everything runs against a fake I wrote. It has not been run against a real property management system, a real browser, a real vision model or any company's agent.
- The burst result says the next thing to build is a second, independent way to check a step (a different screen or report that shows the same change), not more reads of the same screen. I have not built that.
- Misread rates are assumed constant. Real ones would differ by screen and by field, and would need per-step calibration from labelled screenshots.
- The fake applies an action fully or not at all. A real click sequence can stop halfway through a form.
- One worker per task. Two workers running the same task at once would both act. This needs a lease that I did not build.
- The reference tag only works where the screen has a free-text field and a code in it is acceptable. Guest texts still use count mode.
- The journal fsyncs every record. In the benchmark run an append took 6950.1 microseconds with fsync and 24.3 without on this machine.
