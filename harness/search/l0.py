"""L0 search: Optuna over HarnessConfig thresholds, evaluated by the twin on the search split (design doc §5.1).

    python -m harness.search.l0 --manifest data/datasets/manifests/notsofar__sdm.jsonl ... --space routing \
        --trials 60 --name l0_routing_v1          (GPU job: missing decodes are filled as trials need them)

Per trial: config -> twin on every search session -> decodes the cache lacks are filled with the models kept
loaded on the GPU -> twin again (until nothing is missing) -> locked evaluator counts -> Optuna.
Objectives (minimized): cpWER and ORC-WER over all search sessions (Chinese per character).
Constraints (Optuna constraints, <= 0 is feasible): every guard metric of harness/eval/gate.py, overall and per
language, must stay within max(2 % x baseline, floor) of the baseline run (default config), on point values; the
interval-based gate on the select split comes afterwards for the Pareto set only (harness.eval.gate).
Spaces: routing (no new audio spans; only the other engine on existing spans), schedule / activity / vad (new spans:
GPU-heavy), all.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from harness.eval import gate
from harness.eval.metrics import METRICS, session_counts, value
from harness.eval.schema import load_hypotheses, load_references
from harness.twin import config
from harness.twin.fill import Engine, fill, pending
from harness.twin.sim import CostModel
from harness.twin.simulate import H, ROOT, register_audio, run_sessions, sessions_of

SPACES = {
    "routing": {"router.margin": (0.5, 3.0), "router.decideWithin": (1.5, 4.5), "router.speakerWeight": (0.0, 2.5),
                "router.recentWeight": (0.0, 2.0), "router.scriptWeight": (0.0, 2.0),
                "router.rememberEvidence": (0.0, 3.0)},
    "schedule": {"transcription.partialInterval": (0.6, 1.5), "transcription.firstPartialAfter": (0.5, 1.5),
                 "transcription.maximumUtterance": (8.0, 16.0)},
    "activity": {"activity.on": (0.35, 0.7), "activity.off": (0.2, 0.5), "activity.confirm": (0.3, 1.2),
                 "activity.hold": (0.08, 0.6)},
    "vad": {"vad.threshold": (0.3, 0.7), "vad.minSilence": (0.2, 0.8), "vad.minSpeech": (0.1, 0.5)},
}
SPACES["all"] = {k: v for s in SPACES.values() for k, v in s.items()}
OBJECTIVES = ["cp", "orc"]


def apply(cfg, params):
    for k, v in params.items():
        sec, key = k.split(".")
        cfg[sec][key] = v
    if cfg["activity"]["off"] > cfg["activity"]["on"]:
        cfg["activity"]["off"] = cfg["activity"]["on"]
    return cfg


class Evaluator:
    def __init__(self, a, sess, refs):
        self.a, self.sess, self.refs = a, sess, refs
        self.index = json.loads(register_audio(sess).read_text())
        self.engines = {}
        self.cost = CostModel.load(a.cost) if a.cost else None

    def engine(self, name):
        if name not in self.engines:
            ck = {"qwen": "Qwen3-ASR-1.7B", "audar": "Audar-ASR-V1-Turbo"}[name]
            self.engines[name] = Engine(ROOT / "data/checkpoints" / ck, name)
        return self.engines[name]

    def run(self, cfg, out):
        for _ in range(8):
            missing = run_sessions(cfg, self.sess, out, self.cost, quiet=True)
            if not missing:
                break
            for name in ("qwen", "audar"):
                if pending(H / "cache", name):
                    fill(self.engine(name), H / "cache", self.index, quiet=True)
        else:
            raise RuntimeError(f"{out}: decodes still missing after 8 rounds")
        hyps = load_hypotheses(out)
        rows = [session_counts(s, self.refs[s], hyps.get(s)) for s in sorted(self.refs)]
        with open(Path(out) / "counts.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return rows


def groups(rows):
    g = defaultdict(list)
    for r in rows:
        g["all"].append(r)
        g[r["lang"]].append(r)
    return g


def constraints(rows, base_rows):
    out = []
    bg, cg = groups(base_rows), groups(rows)
    for g in bg:
        for m in gate.GUARD:
            b, c = value(bg[g], m), value(cg[g], m)
            if b is None or c is None:
                continue
            tol = max(gate.TOL * abs(b), gate.ABS_FLOOR.get(m, 0.0))
            out.append(((c - b) if METRICS[m][3] else (b - c)) - tol)
    return out


def main():
    import optuna

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--space", default="routing", choices=list(SPACES))
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cost", default=None)
    ap.add_argument("--splits", default=str(ROOT / "harness/splits.json"))
    ap.add_argument("--limit", type=int, default=0, help="debug: first N search sessions only")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sess = sessions_of(a.manifest, "search", a.splits)
    if a.limit:
        sess = dict(sorted(sess.items())[:a.limit])
    refs = {s: r for s, r in load_references(a.manifest).items() if s in sess}
    root = H / "search" / a.name
    root.mkdir(parents=True, exist_ok=True)
    ev = Evaluator(a, sess, refs)
    base_rows = ev.run(config.load(None), root / "baseline")
    base = {m: value(base_rows, m) for m in OBJECTIVES}
    print(f"baseline {base} on {len(sess)} sessions", flush=True)
    space = SPACES[a.space]
    store = {}

    def objective(trial):
        params = {k: trial.suggest_float(k, lo, hi) for k, (lo, hi) in space.items()}
        rows = ev.run(apply(config.load(None), params), root / f"trial_{trial.number:04d}")
        c = constraints(rows, base_rows)
        store[trial.number] = c
        trial.set_user_attr("constraints", c)
        vals = [value(rows, m) for m in OBJECTIVES]
        print(f"trial {trial.number}: {dict(zip(OBJECTIVES, [round(v, 3) for v in vals]))} "
              f"feasible={max(c) <= 0 if c else True}", flush=True)
        return vals

    sampler = optuna.samplers.NSGAIISampler(seed=a.seed,
                                            constraints_func=lambda t: t.user_attrs.get("constraints", [0.0]))
    study = optuna.create_study(directions=["minimize"] * len(OBJECTIVES), sampler=sampler,
                                storage=f"sqlite:///{root}/study.db", study_name=a.name, load_if_exists=True)
    study.enqueue_trial({k: config.load(None)[k.split('.')[0]][k.split('.')[1]] or 0.9 for k in space})
    study.optimize(objective, n_trials=a.trials)
    pareto = [{"number": t.number, "values": t.values, "params": t.params,
               "feasible": max(t.user_attrs.get("constraints", [0])) <= 0} for t in study.best_trials]
    (root / "pareto.json").write_text(json.dumps({"baseline": base, "pareto": pareto}, indent=1))
    print(json.dumps(pareto, indent=1))


if __name__ == "__main__":
    main()
