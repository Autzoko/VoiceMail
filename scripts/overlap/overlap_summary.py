"""Overlap study, step 3: aggregate per-utterance overlap tables into versioned CSVs.

Inputs : data/outputs/analysis/overlap/<model>/<set>.parquet   (overlap_table.py / remix_table.py / libricss)
Outputs: data/results/overlap/<tag>/
    rates.csv        per run x overlap group: rate of every error mode (per 100 ref tokens) + share of the error
                     + utterance probability P(>=1 error of the mode), chance-corrected SWAP / INTRUDE
    token.csv        token error probability: overlapped tokens / non-overlapped tokens of overlapped utterances
                     (spill-over) / tokens of non-overlapped utterances, by SIR bin
    by_factor.csv    overlapped utterances only: rate + utterance probability per factor bin, 95% cluster-bootstrap CI
    regression.csv   overlapped utterances: Poisson GLM rate ratios (offset log n_ref, cluster-robust SE) and
                     logistic odds ratios for P(mode>0), all factors jointly

    python scripts/overlap/overlap_summary.py --tag real      (sets: <ds>__<cond>)
    python scripts/overlap/overlap_summary.py --tag remix --glob 'remix__*'
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

AN = ROOT / "data" / "outputs" / "analysis" / "overlap"
NEAR = {"ihm", "near"}
EN = ["ami", "notsofar", "chime6", "dipco"]
MODES = ["SWAP", "INTRUDE", "OVL_SUB", "OVL_DEL", "OVL_INS", "SUB", "DEL", "INS", "BOUND", "PSEUDO", "HALL", "EMPTY",
         "LID"]
GROUPS = {"overlap": ["SWAP", "INTRUDE", "OVL_SUB", "OVL_DEL", "OVL_INS"], "clean": ["SUB", "DEL", "INS"]}
FACTORS = ["wovl_bin", "sir_bin", "nspk_bin", "position", "int_type", "onset", "ovlsec_bin", "dur_bin", "content"]
REF = {"wovl_bin": "0-0.1", "wovl4": "0-0.1", "sir_bin": ">10", "nspk_bin": "1", "position": "middle", "int_type": "speech",
       "onset": "interrupted", "ovlsec_bin": "<0.5s", "dur_bin": "3-10s", "content": "normal",
       "sir_set": "20", "ratio_set": "0.25", "position_set": "middle", "type": "speech", "n_int": "1"}
REMIX_FACTORS = ["sir_set", "ratio_set", "position_set", "type", "n_int"]


def load(pattern):
    parts = sorted(AN.glob(f"*/{pattern}.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["E"] = df[["S", "D", "I"]].sum(axis=1)
    df["OVERLAP"] = df[GROUPS["overlap"]].sum(axis=1)
    df["CLEAN"] = df[GROUPS["clean"]].sum(axis=1)
    if "dist" not in df:
        df["dist"] = np.where(df.condition.isin(NEAR), "near", "far")
    zh = df.dataset.str.contains("alimeeting")
    df["lang"] = np.where(zh, "zh", "en")
    df["has_ovl"] = df.w_ovl > 0
    df["wovl_bin"] = pd.cut(df.w_ovl, [-1, 0, 0.1, 0.3, 0.5, 0.8, 1.01],
                            labels=["0", "0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.8", ">0.8"]).astype(str)
    # coarser bins for the joint model: position 'full' (w_ovl>=0.8) is then nested in '>=0.5', not collinear
    df["wovl4"] = pd.cut(df.w_ovl, [-1, 0, 0.1, 0.3, 0.5, 1.01], labels=["0", "0-0.1", "0.1-0.3", "0.3-0.5", ">=0.5"],
                         right=False).astype(str)
    df["sir_bin"] = pd.cut(df.sir_db, [-1e9, -5, 0, 5, 10, 1e9], labels=["<-5", "-5-0", "0-5", "5-10", ">10"],
                           right=False).astype(str).replace("nan", "NA")
    df["nspk_bin"] = df.n_int_spk.clip(upper=3).astype(int).astype(str).replace({"3": "3+"})
    df["ovlsec_bin"] = pd.cut(df.ovl_sec, [-1, 0, 0.5, 1, 2, 4, 1e9],
                              labels=["0", "<0.5s", "0.5-1s", "1-2s", "2-4s", ">4s"]).astype(str)
    df["dur_bin"] = pd.cut(df.dur, [0, 1, 3, 10, 20, 1e9], labels=["<1s", "1-3s", "3-10s", "10-20s", ">20s"],
                           right=False).astype(str)
    small, short = np.where(zh, 3, 2), np.where(zh, 8, 5)
    df["content"] = np.select([df.n_ref <= small, df.n_ref <= short], ["backchannel", "short"], "normal")
    base = df.session.astype(str).str.split("_sc_").str[0]
    df["cluster"] = df.dataset + "/" + base + "/" + df.speaker.astype(str)
    if "cluster_id" in df:  # remix: cluster = source target segment's speaker
        df["cluster"] = df.cluster_id
    return df


def run_groups(df):
    """(model, dataset, dist) groups incl. pooled English."""
    out = [((m, d, dist), s) for (m, d, dist), s in df.groupby(["model", "dataset", "dist"])]
    en = df[df.lang == "en"]
    if en.dataset.nunique() > 1:
        out += [((m, "en-all", dist), s) for (m, dist), s in en.groupby(["model", "dist"])]
    return out


def rates(df):
    rows = []
    cats = ["E", "OVERLAP", "CLEAN"] + MODES
    for key, s in run_groups(df):
        for grp, g in [("no_overlap", s[~s.has_ovl]), ("overlap", s[s.has_ovl])]:
            if not len(g):
                continue
            nref, E = g.n_ref.sum(), g.E.sum()
            r = dict(zip(["model", "dataset", "dist"], key), group=grp, n_utts=len(g), n_ref=int(nref))
            for c in cats:
                r[f"{c}_rate"] = 100 * g[c].sum() / nref
                r[f"{c}_share"] = 100 * g[c].sum() / max(E, 1)
                r[f"{c}_p_utt"] = 100 * (g[c] > 0).mean()
            for c in ("SWAP", "INTRUDE"):
                r[f"{c}_ctrl_rate"] = 100 * g[f"{c}_ctrl"].sum() / nref
                r[f"{c}_corr_rate"] = r[f"{c}_rate"] - r[f"{c}_ctrl_rate"]
            r["target_lost_p_utt"] = 100 * ((g.err_ovl_tok >= 0.5 * g.n_ovl_tok) & (g.n_ovl_tok > 0)).mean()
            rows.append(r)
    return pd.DataFrame(rows).round(3)


def token(df):
    rows = []
    for key, s in run_groups(df):
        for sir, g in [("all", s)] + list(s[s.has_ovl].groupby("sir_bin")):
            o = g[g.has_ovl]
            n_ovl, e_ovl = o.n_ovl_tok.sum(), o.err_ovl_tok.sum()
            n_cl, e_cl = (o.n_ref - o.n_ovl_tok).sum(), o.err_clean_tok.sum()
            no = s[~s.has_ovl]
            r = dict(zip(["model", "dataset", "dist"], key), sir_bin=sir,
                     p_err_overlapped_tok=100 * e_ovl / max(n_ovl, 1),
                     p_err_clean_tok_in_ovl_utt=100 * e_cl / max(n_cl, 1),
                     p_err_tok_no_ovl_utt=100 * no.err_clean_tok.sum() / max(no.n_ref.sum(), 1) if sir == "all" else np.nan,
                     n_overlapped_tok=int(n_ovl), n_clean_tok=int(n_cl))
            rows.append(r)
    return pd.DataFrame(rows).round(3)


def _boot(sub, factor, cats, n_boot=300, seed=0):
    piv = sub.groupby(["cluster", factor])[["n_ref"] + cats].sum()
    pu = sub.assign(**{f"u_{c}": (sub[c] > 0).astype(int) for c in cats}, one=1)
    piu = pu.groupby(["cluster", factor])[["one"] + [f"u_{c}" for c in cats]].sum()
    clusters = piv.index.get_level_values(0).unique()
    bins = piv.index.get_level_values(1).unique()
    mi = pd.MultiIndex.from_product([clusters, bins])
    arr = piv.reindex(mi, fill_value=0).to_numpy().reshape(len(clusters), len(bins), -1)
    aru = piu.reindex(mi, fill_value=0).to_numpy().reshape(len(clusters), len(bins), -1)
    w = np.random.default_rng(seed).multinomial(len(clusters), [1 / len(clusters)] * len(clusters), size=n_boot)
    tot, totu = np.einsum("rc,cbk->rbk", w, arr), np.einsum("rc,cbk->rbk", w, aru)
    rate = 100 * tot[:, :, 1:] / np.maximum(tot[:, :, :1], 1)
    pr = 100 * totu[:, :, 1:] / np.maximum(totu[:, :, :1], 1)
    pt = 100 * arr.sum(0)[:, 1:] / np.maximum(arr.sum(0)[:, :1], 1)
    ptu = 100 * aru.sum(0)[:, 1:] / np.maximum(aru.sum(0)[:, :1], 1)
    lo, hi = np.percentile(rate, [2.5, 97.5], axis=0)
    lou, hiu = np.percentile(pr, [2.5, 97.5], axis=0)
    rows = []
    for bi, b in enumerate(bins):
        for ki, k in enumerate(cats):
            rows.append({"factor": factor, "bin": b, "n_utts": int(aru.sum(0)[bi, 0]), "n_ref": int(arr.sum(0)[bi, 0]),
                         "category": k, "rate": pt[bi, ki], "lo": lo[bi, ki], "hi": hi[bi, ki],
                         "p_utt": ptu[bi, ki], "p_lo": lou[bi, ki], "p_hi": hiu[bi, ki]})
    return rows


def by_factor(df, factors):
    cats = ["E", "OVERLAP", "CLEAN"] + MODES[:9]
    out = []
    for key, s in run_groups(df):
        o = s[s.has_ovl]
        if len(o) < 50:
            continue
        for f in factors:
            if f not in o or o[f].nunique() < 2:
                continue
            for r in _boot(o, f, cats):
                out.append(dict(zip(["model", "dataset", "dist"], key), **r))
    return pd.DataFrame(out).round(3)


def regression(df, factors):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    rows = []
    cats = ["E", "OVERLAP", "SWAP", "INTRUDE", "OVL_SUB", "OVL_DEL", "OVL_INS", "CLEAN", "BOUND", "HALL"]
    for key, s in run_groups(df):
        if key[1] not in ("en-all", "alimeeting", "remix_alimeeting") and not key[1].startswith("libricss"):
            continue
        o = s[s.has_ovl & (s.n_ref > 0)]
        facs = [("wovl4" if f == "wovl_bin" else f) for f in factors if f != "ovlsec_bin"]
        facs = [f for f in facs if f in o and o[f].nunique() > 1]
        if "sir_bin" in facs:
            o = o[o.sir_bin != "NA"]
        o = o.dropna(subset=facs + ["cluster"])
        extra = "C(dataset) + " if key[1] == "en-all" else ""
        rhs = " + ".join(f"C({f}, Treatment('{REF[f]}'))" if REF.get(f) in set(o[f]) else f"C({f})" for f in facs)
        for c in cats:
            if o[c].sum() < 30:
                continue
            for kind in ("rate", "p_utt"):
                d = o.assign(y=(o[c] > 0).astype(int)) if kind == "p_utt" else o
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if kind == "rate":
                            m = smf.glm(f"{c} ~ {extra}{rhs}", data=d, family=sm.families.Poisson(),
                                        offset=np.log(d.n_ref))
                        else:
                            m = smf.glm(f"y ~ {extra}{rhs}", data=d, family=sm.families.Binomial())
                        m = m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.cluster)[0]})
                except Exception as e:  # noqa: BLE001
                    print(f"fit failed {key} {c} {kind}: {e}")
                    continue
                ci = m.conf_int()
                for term in m.params.index:
                    if term == "Intercept" or term.startswith("C(dataset"):
                        continue
                    f = term.split(",")[0].split(")")[0][2:]
                    lvl = term.split("[T.")[-1].rstrip("]")
                    est, lo, hi = np.exp(m.params[term]), np.exp(ci.loc[term, 0]), np.exp(ci.loc[term, 1])
                    n_lvl = int((d[f].astype(str) == lvl).sum())
                    ev_lvl = int((d.loc[d[f].astype(str) == lvl, c] > 0).sum())
                    rows.append({"model": key[0], "dataset": key[1], "dist": key[2], "category": c, "kind": kind,
                                 "factor": f, "level": lvl, "ref_level": REF.get(f, ""),
                                 "ratio": est, "lo": lo, "hi": hi, "p": m.pvalues[term],
                                 "n_utts_level": n_lvl, "n_utts_with_event_level": ev_lvl,
                                 # quasi-separation: too few events in the level or an unbounded CI
                                 "valid": bool(ev_lvl >= 5 and np.isfinite(hi) and hi / max(lo, 1e-300) < 1e4)})
    r = pd.DataFrame(rows)
    return r.round(4) if len(r) else r


def paired(df):
    """Remix: change of every error mode vs the same target decoded alone (per 100 target tokens), by design
    factor; noise twins vs their speech counterparts. 95% CI by cluster bootstrap."""
    cats = ["E", "OVERLAP", "CLEAN"] + MODES[:9]
    clean = df[df.variant == "clean"].set_index(["model", "target_id"])[cats]
    m = df[df.variant != "clean"].join(clean, on=["model", "target_id"], rsuffix="_clean")
    for c in cats:
        m[f"d_{c}"] = m[c] - m[f"{c}_clean"]
    m["arm"] = np.where(m.noise_twin.astype(bool), "noise_twin", "overlap")
    dc = [f"d_{c}" for c in cats]
    out = []
    for key, s in run_groups(m):
        for f in ["arm", "sir_set", "ratio_set", "position_set", "type", "n_int"]:
            sub = s if f == "arm" else s[s.arm == "overlap"]
            for r in _boot(sub, f, dc):
                out.append(dict(zip(["model", "dataset", "dist"], key), **r))
        tw = s[s.arm == "noise_twin"]
        for r in _boot(tw, "sir_set", dc):
            out.append(dict(zip(["model", "dataset", "dist"], key), **{**r, "factor": "noise_twin_by_sir"}))
    o = pd.DataFrame(out).drop(columns=["p_utt", "p_lo", "p_hi"], errors="ignore")
    return o.round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="real")
    ap.add_argument("--glob", default="[a-z]*__[a-z]*")
    ap.add_argument("--factors", nargs="+", default=FACTORS)
    args = ap.parse_args()
    out = ROOT / "data" / "results" / "overlap" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.glob)
    if args.tag == "real":
        df = df[~df.dataset.str.startswith(("remix", "libricss"))]
    if args.tag == "remix":
        df = df[df.dataset.str.startswith("remix")]
        df["has_ovl"] = (df.variant != "clean") & ~df.noise_twin.astype(bool)
        for c in ("ratio_set", "n_int"):
            df[c] = df[c].astype(str)
        args.factors = REMIX_FACTORS + ["wovl_bin", "onset"]
        paired(df).to_csv(out / "paired.csv", index=False)
    # remix: backchannel <=> ratio 0 <=> position 'pause' are collinear -> the joint model uses speech mixes only
    reg_df = df[df.type == "speech"] if args.tag == "remix" else df
    reg_f = ["sir_set", "ratio_set", "position_set", "n_int"] if args.tag == "remix" else args.factors
    print(f"{len(df)} utterance rows", flush=True)
    for name, fn in [("rates", lambda: rates(df)), ("token", lambda: token(df)),
                     ("by_factor", lambda: by_factor(df, args.factors)),
                     ("regression", lambda: regression(reg_df, reg_f))]:
        t = fn()
        t.to_csv(out / f"{name}.csv", index=False)
        print(f"  {name}: {len(t)} rows", flush=True)
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
