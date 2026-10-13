#!/usr/bin/env python3
"""Paired LoRA-vs-control comparison with the pre-registered GO/NO-GO rule (plan Stage 7).

Why paired runs
    A dev score moves by about +-1.5 tasks from luck alone (sampling temperature, timeouts), so one
    LoRA run against one control run says little. Stage 7 runs both arms in the same notebook, on the
    same server and the same tasks, in pairs (lora1 vs ctl1, lora2 vs ctl2, ...). Pairing cancels
    whatever both arms share (task difficulty, server speed), leaving mostly the adapter's effect.

The statistics, in plain words
    A "cell" is one task in one pair: (pair k, task t). Each cell has four possible outcomes:
        both pass, both fail      concordant: says nothing about which arm is better
        LoRA pass, control fail   counted in b
        LoRA fail, control pass   counted in c
    Only the discordant cells (b + c of them) carry evidence. If the adapter made no difference
    (the "null hypothesis"), each discordant cell would be a fair coin flip between "LoRA won" and
    "control won", so b would follow Binomial(n = b + c, p = 0.5).

    McNemar's exact test asks: if it really were a fair coin, how likely is a result at least as
    good for LoRA as the one we saw? That probability is the p-value:
        p = P(Binomial(b + c, 0.5) >= b) = sum_{k=b..b+c} C(b+c, k) / 2^(b+c)
    It is one-sided because we only act on an improvement. It is "exact" because it sums the
    binomial terms directly instead of using the chi-square approximation, which is poor for the
    small counts we have. Examples: b=6, c=0 gives p = 1/64 = 0.016; b=7, c=1 gives 9/256 = 0.035;
    b=8, c=2 gives 56/1024 = 0.055 (not significant at 0.05). Identical runs (b = c = 0) give p = 1.

    p <= 0.05 means a no-effect adapter would pass this bar at most 5% of the time (the false-GO
    rate). A small p alone is not enough: with many cells a tiny gain can be "significant", so the
    rule also asks for an effect size, the mean per-pair delta (LoRA resolved minus control
    resolved, which equals (b - c) / pairs) of at least +1.5 tasks per run.

    Caveat: McNemar treats every cell as independent, but the same task appears in every pair and
    some tasks are just easier for both arms. The p-value is therefore a little optimistic, which is
    one more reason the effect-size bar exists.

Pre-registered rule (plan Stage 7; thresholds fixed before seeing any data, so we cannot move them)
    E1, after 2 pairs: stop early (NO-GO, skip E2) if any of
        summed delta over pairs 1-2 <= -2; any LoRA arm loses more than 1 core task;
        any overflow, crash or undeclared-tool error in a LoRA arm; LoRA s/task > 1.15 x control s/task.
    E2, after 4 pairs: GO needs all of
        mean delta >= +1.5; exact one-sided McNemar p <= 0.05; LoRA keeps core in all but at most 1
        core cell (23 of 24); malformed calls <= 2 in every LoRA run; repeat% <= 6 in every LoRA run;
        overflow 0; projected scorer runtime <= 10.5 h. Crashes and undeclared-tool errors stay vetoes
        (carried over from E1, since E2 runs could show them for the first time), and E2 needs 4 pairs.
    The projected runtime scales the Stage 2 full-dev run (--base-run) by the LoRA/control s/task
    ratio, with the Stage 2 formula 120 x (hours / tasks) + 0.2.
    The vetoes look at the LoRA arms. A crash or overflow in a control arm does not veto the adapter
    (the prompt-only config we would keep has the same problem), but it is printed as a WARNING
    because it makes that pair's comparison suspect.
    A crash is a task that is missing from task_results.jsonl or ended with an error that is not a
    normal budget ending (timeout, turn/tool budget, no submit_patch, test_patch did not apply).

Usage
    python scripts/paired_compare.py lora1:ctl1 lora2:ctl2 --runs-dir eval/runs/e1/runs --tasks dev_dec
    python scripts/paired_compare.py lora1:ctl1 lora2:ctl2 lora3:ctl3 lora4:ctl4 \\
        --runs-dir eval/runs/e1/runs --runs-dir eval/runs/e2/runs --tasks dev_dec --base-run eval/runs/t0/runs/think
    python scripts/paired_compare.py --pair path/to/lora1 path/to/ctl1 --pair path/to/lora2 path/to/ctl2
Each side of LORA:CTL is a run directory, a variant name found under one of the --runs-dir folders,
or LABEL=DIR. --pair LORA CTL (repeatable) takes two paths and avoids splitting on ':'.
A run directory holds results/task_results.jsonl (and traces/, patches/) as written by agent_eval.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import load_run, summarize  # noqa: E402  (scripts/ is not a package)

ROOT = Path(__file__).resolve().parents[1]

# Pre-registered thresholds (plan Stage 7). Change them only before a run, never after.
EARLY_SUM_DELTA_MAX = -2       # E1 stops if summed delta over pairs 1-2 is <= this
CORE_MISSES_PER_ARM = 1        # E1: each LoRA arm keeps >= 5/6 core
SPEED_RATIO_MAX = 1.15         # E1: LoRA s/task <= 1.15 x control s/task
GO_PAIRS = 4                   # E2: GO is judged on all 4 pairs (E1 + E2), never fewer
GO_MEAN_DELTA = 1.5            # E2: tasks per run
GO_P = 0.05
GO_CORE_MISSES = 1             # E2: 23 of 24 core cells
GO_MALFORMED_PER_RUN = 2
GO_REPEAT_PCT = 6.0
GO_RUNTIME_H = 10.5
HIDDEN_TASKS, STARTUP_H = 120, 0.2  # Stage 2 projection: 120 x (hours / tasks) + 0.2

# Task errors that are normal endings, not crashes. Anything else in the error field is a crash.
BENIGN_ERRORS = ("session timeout", "turns budget", "tool call budget", "without calling submit_patch",
                 "Failed to apply test_patch")


def mcnemar_p(b: int, c: int) -> float:
    """Exact one-sided McNemar p = P(Binomial(b + c, 0.5) >= b). Integer sums, so no float error."""
    n = b + c
    return sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n


def resolve_run(spec: str, runs_dirs: list[Path]) -> Path:
    """A run is a directory, LABEL=DIR, or a variant name under one of runs_dirs (E1 and E2 are separate notebooks)."""
    _, sep, path = spec.partition("=")
    if sep and Path(path).is_dir():
        return Path(path)
    if Path(spec).is_dir():
        return Path(spec)
    found = [d / spec for d in runs_dirs if (d / spec).is_dir()]
    if len(found) != 1:
        raise SystemExit(f"run {spec!r}: expected one match under {[str(d) for d in runs_dirs]}, found {found}")
    return found[0]


def is_undeclared(error: str) -> bool:
    """ADK's error when the model calls a tool the agent does not declare ("Tool 'grep_rnI' not found")."""
    return "Tool '" in error and "not found" in error


def is_crash(error: str, overflow: bool) -> bool:
    """An unexpected task ending. Overflows and undeclared tools are counted separately, so not here."""
    return bool(error) and not overflow and not is_undeclared(error) and not any(b in error for b in BENIGN_ERRORS)


def run_errors(run: Path) -> dict[str, str]:
    """instance_id -> error text (compare_runs keeps only flags derived from it). No file -> no tasks."""
    path = run / "results" / "task_results.jsonl"
    out: dict[str, str] = {}
    if not path.exists():  # the notebook died before writing results: every task counts as a crash
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["instance_id"]] = r.get("error") or ""
    return out


def arm_stats(run: Path, task_ids: list[str], fairness: dict) -> dict:
    """compare_runs metrics on the task set, plus crashes (unknown errors or missing tasks)."""
    tasks = load_run(run) if (run / "results" / "task_results.jsonl").exists() else {}
    missing = [t for t in task_ids if t not in tasks]
    kept = {t: tasks[t] for t in task_ids if t in tasks}
    errors = run_errors(run)
    s = summarize(kept, fairness, max_minutes=5.0) if kept else {"resolved": 0, "core_kept": 0, "malformed": 0,
                                                                  "repeat_pct": 0.0, "overflow": 0,
                                                                  "undeclared_tool": 0, "mean_s": 0.0}
    invalid = set(missing) | {t for t in kept if is_crash(errors.get(t, ""), kept[t]["overflow"])}
    return {
        "run": str(run), "passed": {t: kept.get(t, {}).get("resolved", False) for t in task_ids},
        "invalid": sorted(invalid),
        "resolved": s["resolved"], "core_kept": s["core_kept"], "malformed": s["malformed"],
        "repeat_pct": s["repeat_pct"], "overflow": s["overflow"], "mean_s": s["mean_s"],
        "seconds": sum(t["seconds"] for t in kept.values()),
        "undeclared": s["undeclared_tool"] + sum(is_undeclared(errors.get(t, "")) for t in kept),
        "crashes": len(missing) + sum(is_crash(errors.get(t, ""), kept[t]["overflow"]) for t in kept),
        "missing": missing,
    }


def check(name: str, ok: bool | None, value: str, rule: str) -> dict:
    return {"check": name, "ok": ok, "value": value, "rule": rule}


def compare(pairs: list[tuple[Path, Path]], task_ids: list[str], fairness: dict, stage: str = "auto",
            base_run: Path | None = None) -> dict:
    """Score the pairs and apply the Stage 7 rule. stage 'e1' = early-stop check, 'e2' = GO check."""
    core = [t for t in fairness.get("core", []) if t in task_ids]
    stage = stage if stage != "auto" else ("e2" if len(pairs) >= 4 else "e1")
    if not task_ids:
        raise ValueError("no tasks to compare: did the notebooks die before writing any results?")
    rows, b, c = [], 0, 0
    for k, (lora_dir, ctl_dir) in enumerate(pairs, 1):
        lora, ctl = arm_stats(lora_dir, task_ids, fairness), arm_stats(ctl_dir, task_ids, fairness)
        # A cell where either arm crashed or is missing says nothing about the adapter: a dead control
        # would otherwise turn every task the adapter solved into a fake "adapter only" win.
        usable = [t for t in task_ids if t not in lora["invalid"] and t not in ctl["invalid"]]
        bk = sum(lora["passed"][t] and not ctl["passed"][t] for t in usable)
        ck = sum(ctl["passed"][t] and not lora["passed"][t] for t in usable)
        b, c = b + bk, c + ck
        delta = sum(lora["passed"][t] for t in usable) - sum(ctl["passed"][t] for t in usable)
        rows.append({"pair": k, "lora": lora, "ctl": ctl, "delta": delta, "b": bk, "c": ck})

    loras, ctls = [r["lora"] for r in rows], [r["ctl"] for r in rows]
    mean_delta = sum(r["delta"] for r in rows) / len(rows)
    p = mcnemar_p(b, c)
    # Pooled s/task: total seconds over all cells, so one slow pair weighs as much as its tasks.
    ratio = (sum(a["seconds"] for a in loras) / max(1e-9, sum(a["seconds"] for a in ctls)))
    core_misses = [len(core) - a["core_kept"] for a in loras]
    safety = [
        check("LoRA overflow", sum(a["overflow"] for a in loras) == 0, str([a["overflow"] for a in loras]), "0"),
        check("LoRA crashes", sum(a["crashes"] for a in loras) == 0, str([a["crashes"] for a in loras]), "0"),
        check("LoRA undeclared tool", sum(a["undeclared"] for a in loras) == 0,
              str([a["undeclared"] for a in loras]), "0"),
    ]
    if stage == "e1":
        early = rows[:2]
        checks = [
            check("summed delta pairs 1-2", sum(r["delta"] for r in early) > EARLY_SUM_DELTA_MAX,
                  str(sum(r["delta"] for r in early)), f"> {EARLY_SUM_DELTA_MAX}"),
            check("LoRA core misses per arm", max(core_misses) <= CORE_MISSES_PER_ARM,
                  f"{core_misses} of {len(core)}", f"<= {CORE_MISSES_PER_ARM} each"),
            *safety,
            check("LoRA/control s/task", ratio <= SPEED_RATIO_MAX, f"{ratio:.2f}", f"<= {SPEED_RATIO_MAX}"),
        ]
        verdict = "CONTINUE (run E2)" if all(x["ok"] for x in checks) else "NO-GO (early stop)"
    else:
        runtime = None
        if base_run is not None:
            base = json.loads((base_run / "run_summary.json").read_text())
            runtime = HIDDEN_TASKS * base["hours"] * ratio / base["total"] + STARTUP_H
        checks = [
            check("pairs", len(rows) >= GO_PAIRS, str(len(rows)), f">= {GO_PAIRS}"),
            check("mean delta", mean_delta >= GO_MEAN_DELTA, f"{mean_delta:+.2f}", f">= +{GO_MEAN_DELTA}"),
            check("McNemar p (one-sided)", p <= GO_P, f"{p:.4f}", f"<= {GO_P}"),
            check("LoRA core cells lost", sum(core_misses) <= GO_CORE_MISSES,
                  f"{sum(core_misses)} of {len(core) * len(loras)}", f"<= {GO_CORE_MISSES}"),
            check("malformed per LoRA run", max(a["malformed"] for a in loras) <= GO_MALFORMED_PER_RUN,
                  str([a["malformed"] for a in loras]), f"<= {GO_MALFORMED_PER_RUN}"),
            check("repeat% per LoRA run", max(a["repeat_pct"] for a in loras) <= GO_REPEAT_PCT,
                  str([a["repeat_pct"] for a in loras]), f"<= {GO_REPEAT_PCT}"),
            *safety,
            check("projected scorer h", None if runtime is None else runtime <= GO_RUNTIME_H,
                  "no --base-run" if runtime is None else f"{runtime:.1f} (s/task ratio {ratio:.2f})",
                  f"<= {GO_RUNTIME_H}"),
        ]
        if any(x["ok"] is False for x in checks):
            verdict = "NO-GO"
        else:
            verdict = "GO" if all(x["ok"] for x in checks) else "UNDECIDED (pass --base-run for the runtime check)"
    # A broken control arm does not count against the adapter, but the comparison is then not trustworthy:
    # never answer GO / CONTINUE from it. Rerun the pair.
    broken = [r["pair"] for r in rows if r["ctl"]["invalid"]]
    if broken and all(x["ok"] for x in safety):  # an adapter-side safety veto stands on its own
        verdict = f"INVALID (control arm incomplete or crashed in pair(s) {broken}: rerun them)"
    warnings = [f"pair {r['pair']} control {r['ctl']['run']}: {r['ctl']['crashes']} crashes, "
                f"{r['ctl']['overflow']} overflows, missing {r['ctl']['missing']}"
                for r in rows if r["ctl"]["crashes"] or r["ctl"]["overflow"] or r["ctl"]["invalid"]]
    return {"stage": stage, "tasks": task_ids, "pairs": rows, "b": b, "c": c, "mean_delta": mean_delta,
            "p": p, "s_per_task_ratio": ratio, "checks": checks, "verdict": verdict, "warnings": warnings}


def print_report(rep: dict, fairness: dict) -> None:
    label = {t: k for k in ("core", "fixable", "unfair") for t in fairness.get(k, [])}
    rows = rep["pairs"]
    print(f"{len(rep['tasks'])} tasks x {len(rows)} pairs, stage {rep['stage'].upper()}")
    for r in rows:
        print(f"  pair {r['pair']}: LoRA {r['lora']['run']}  vs  control {r['ctl']['run']}")
    print()
    print(f"{'task':16} {'label':8} {'LoRA':>5} {'ctl':>5}")
    for t in rep["tasks"]:
        lp = sum(r["lora"]["passed"][t] for r in rows)
        cp = sum(r["ctl"]["passed"][t] for r in rows)
        print(f"{t:16} {label.get(t, ''):8} {lp:>3}/{len(rows)} {cp:>3}/{len(rows)}" + ("  <" if lp != cp else ""))
    print(f"\n{'pair':5} {'LoRA':>5} {'ctl':>5} {'delta':>6} {'b':>3} {'c':>3}  "
          f"{'core':>9} {'malformed':>9} {'repeat%':>9} {'overflow':>9} {'s/task':>11}")
    for r in rows:
        lo, ct = r["lora"], r["ctl"]
        print(f"{r['pair']:<5} {lo['resolved']:>5} {ct['resolved']:>5} {r['delta']:>+6} {r['b']:>3} {r['c']:>3}  "
              + " ".join(f"{str(lo[k]) + '/' + str(ct[k]):>9}" for k in ("core_kept", "malformed", "repeat_pct", "overflow"))
              + f" {lo['mean_s']:>5.0f}/{ct['mean_s']:<5.0f}")
    print("(metric columns are LoRA/control)")
    print(f"\nmean delta {rep['mean_delta']:+.2f} tasks/run   discordant b={rep['b']} c={rep['c']}   "
          f"McNemar one-sided p={rep['p']:.4f}   LoRA/control s/task {rep['s_per_task_ratio']:.2f}\n")
    for x in rep["checks"]:
        mark = {True: "ok  ", False: "FAIL", None: "??  "}[x["ok"]]
        print(f"  {mark} {x['check']:26} {x['value']:30} rule {x['rule']}")
    for w in rep["warnings"]:
        print(f"WARNING {w}")
    print(f"\nVERDICT: {rep['verdict']}")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pairs", nargs="*", metavar="LORA:CTL", help="run dirs or variant names, LoRA first")
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("LORA_DIR", "CTL_DIR"),
                        help="one pair as two paths (repeatable); pairs are numbered positional first")
    parser.add_argument("--runs-dir", type=Path, action="append", default=[],
                        help="folder holding variant run dirs (repeatable: E1 and E2 notebooks)")
    parser.add_argument("--tasks", help="task set name in --sets-file (e.g. dev_dec, dev_fair); default all tasks seen")
    parser.add_argument("--sets-file", type=Path, default=ROOT / "eval" / "decision_sets.json")
    parser.add_argument("--fairness", type=Path, default=ROOT / "eval" / "fairness.json")
    parser.add_argument("--stage", choices=["auto", "e1", "e2"], default="auto",
                        help="e1 = early-stop rule, e2 = GO rule; auto = e2 from 4 pairs on")
    parser.add_argument("--base-run", type=Path, help="Stage 2 full-dev run dir (run_summary.json) for the runtime projection")
    parser.add_argument("--json", type=Path, help="write the full report here")
    args = parser.parse_args(argv)

    pairs = []
    for spec in args.pairs:
        lora, sep, ctl = spec.partition(":")
        if not sep or not lora or not ctl:
            parser.error(f"pair {spec!r} is not LORA:CTL")
        pairs.append((resolve_run(lora, args.runs_dir), resolve_run(ctl, args.runs_dir)))
    pairs += [(resolve_run(lora, args.runs_dir), resolve_run(ctl, args.runs_dir)) for lora, ctl in args.pair]
    if not pairs:
        parser.error("give at least one LORA:CTL or --pair LORA_DIR CTL_DIR")
    fairness = json.loads(args.fairness.read_text()) if args.fairness.exists() else {}
    if args.tasks:
        sets = json.loads(args.sets_file.read_text())
        if args.tasks not in sets:
            parser.error(f"--tasks {args.tasks!r} not in {args.sets_file} (has {[k for k in sets if k != '_about']})")
        task_ids = sets[args.tasks]
    else:  # every task any run attempted; a task missing from a run then counts as a crash there
        task_ids = sorted({t for pair in pairs for run in pair for t in run_errors(run)})

    if not task_ids:
        parser.error("no task results in any run (did the notebooks die before finishing a task?) and no --tasks set")
    if not fairness.get("core"):
        parser.error(f"{args.fairness} is missing or has no 'core' tasks: the core-kept checks would pass vacuously")
    rep = compare(pairs, task_ids, fairness, args.stage, args.base_run)
    print_report(rep, fairness)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    main()
