"""Summarise a steplog journal: what each kind of step cost and where tasks
stopped. This is the observability side of the runner. Everything it prints
comes from the journal file, so it works on a journal from a dead worker.

    python report.py journal.jsonl
"""
import sys

from steplog import Journal


def summarise(records):
    """-> dict with per-step-name stats, escalation reasons, open intents and
    the done records that were accepted with the weakest evidence. "contested"
    counts done records whose final check had at least one read saying no.
    Reads are those of the final check of each step, which is what the done
    record stores."""
    names, steps, esc, weakest = {}, {}, {}, []
    tasks, open_intents = set(), {}
    for r in records:
        t, i, k = r.get("t"), r.get("i"), r.get("k")
        tasks.add(t)
        if k == "intent":
            names[(t, i)] = r.get("n", "?")
            open_intents[(t, i)] = True
            continue
        name = names.get((t, i), "?")
        s = steps.setdefault(name, dict(done=0, acted=0, found=0, escalated=0,
                                        reads=0, checks=0, contested=0))
        if k == "done":
            open_intents.pop((t, i), None)
            s["done"] += 1
            s[r.get("how", "acted")] = s.get(r.get("how", "acted"), 0) + 1
            if "reads" in r:
                s["reads"] += r["reads"]
                s["checks"] += 1
                s["contested"] += 1 if r.get("no", 0) else 0
            if "p" in r:
                weakest.append((r["p"], t, i, name, r.get("yes", 0), r.get("no", 0)))
        elif k == "escalated":
            open_intents.pop((t, i), None)
            s["escalated"] += 1
            esc[r.get("why", "?")] = esc.get(r.get("why", "?"), 0) + 1
    weakest.sort()
    return dict(tasks=len(tasks), steps=steps, escalations=esc,
                open_intents=sorted("%s step %s (%s)" % (t, i, names[(t, i)])
                                    for t, i in open_intents),
                weakest=weakest[:5])


def render(summary):
    out = ["tasks: %d" % summary["tasks"],
           "%-20s %5s %6s %6s %10s %17s %10s" % ("step", "done", "acted", "found", "escalated",
                                               "reads/final check", "contested")]
    for name in sorted(summary["steps"]):
        s = summary["steps"][name]
        rpc = "%.2f" % (s["reads"] / s["checks"]) if s["checks"] else "n/a"
        out.append("%-20s %5d %6d %6d %10d %17s %10d" % (
            name[:20], s["done"], s["acted"], s["found"], s["escalated"], rpc, s["contested"]))
    out.append("escalations: " + (", ".join("%s x%d" % kv for kv in sorted(summary["escalations"].items()))
                                  or "none"))
    out.append("open intents (worker died mid-step, next run will look before acting): "
               + (", ".join(summary["open_intents"]) or "none"))
    if summary["weakest"]:
        out.append("done records accepted with the least certainty:")
        for p, t, i, name, yes, no in summary["weakest"]:
            out.append("  %s step %s %s p=%.6f  reads said yes %d, no %d" % (t, i, name, p, yes, no))
    return "\n".join(out)


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip())
        return 2
    j = Journal(argv[1])
    j.close()
    print(render(summarise(j.records)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
