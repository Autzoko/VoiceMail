"""Phase 1 error analysis, step 2: aggregate the per-utterance tables into versioned CSVs + figures.

Inputs : data/outputs/analysis/utt/<model>/*.parquet (build_utt_table.py)
Outputs: data/results/error_analysis/
    breakdown.csv   WER/CER of every run decomposed into error types (percentage points + share), checked
                    against data/results/qwen3asr_baseline.csv
    by_factor.csv   per error type: rate vs condition bins (overlap, duration, SNR, content, speaking rate,
                    AMI native language), 95% CI by cluster bootstrap over (meeting, speaker)
    regression.csv  Poisson GLM per error type with offset log(n_ref) and cluster-robust SE (meeting x speaker):
                    rate ratios of each factor while controlling for the others; near and far fitted separately
    compare.csv     near->far, auto->forced, 0.6B->1.7B change of every error type (pp)
    figs/*.png

    python scripts/analysis/summarize.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.error_taxonomy import CATS  # noqa: E402

OUT = ROOT / "data" / "results" / "error_analysis"
NEAR = {"ihm", "near"}
ORDER = ["ami", "alimeeting", "notsofar", "chime6", "dipco"]
EN = ["ami", "notsofar", "chime6", "dipco"]
FACTORS = ["ovl_bin", "dur_bin", "snr_bin", "content", "rate_bin", "native"]
REF_LEVEL = {"ovl_bin": "0", "dur_bin": "3-10s", "snr_bin": ">=20", "content": "normal", "rate_bin": "q3",
             "native": "native"}
RUN = ["model", "lang_mode", "dataset", "condition"]


def load():
    parts = sorted((ROOT / "data/outputs/analysis/utt").glob("*/*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["E"] = df[["S", "D", "I"]].sum(axis=1)
    df["ACOUS"] = df[["ACOUS_S", "ACOUS_D", "ACOUS_I"]].sum(axis=1)
    df["dist"] = np.where(df.condition.isin(NEAR), "near", "far")
    zh = df.dataset == "alimeeting"
    df["ovl_bin"] = pd.cut(df.overlap, [-1, 0, 0.3, 0.7, 1.01], labels=["0", "0-0.3", "0.3-0.7", ">0.7"]).astype(str)
    df["dur_bin"] = pd.cut(df.dur, [0, 1, 3, 10, 20, 1e9], labels=["<1s", "1-3s", "3-10s", "10-20s", ">20s"],
                           right=False).astype(str)
    df["snr_bin"] = pd.cut(df.snr_db, [-1e9, 5, 10, 15, 20, 1e9], labels=["<5", "5-10", "10-15", "15-20", ">=20"],
                           right=False).astype(str).replace("nan", "NA")
    small, short = np.where(zh, 3, 2), np.where(zh, 8, 5)
    df["content"] = np.select([df.n_ref <= small, df.n_ref <= short], ["backchannel", "short"], "normal")
    df["rate"] = df.n_ref / df.dur
    df["rate_bin"] = "NA"
    for is_zh in (True, False):  # speaking-rate quintiles per language (tokens/s; chars for zh)
        m = zh == is_zh
        df.loc[m, "rate_bin"] = pd.qcut(df.loc[m, "rate"], 5, labels=["q1", "q2", "q3", "q4", "q5"]).astype(str)
    base = df.session.str.split("_sc_").str[0]
    df["cluster"] = df.dataset + "/" + base + "/" + df.speaker.astype(str)
    return df


def breakdown(df):
    g = df.groupby(RUN)
    b = g[["n_ref", "E", "xtalk_ctrl"] + CATS].sum()
    b.insert(0, "n_utts", g.size())
    out = b[["n_utts", "n_ref"]].copy()
    out["error_rate"] = 100 * b.E / b.n_ref
    for c in CATS:
        out[c] = 100 * b[c] / b.n_ref
    for c in CATS:
        out[f"share_{c}"] = 100 * b[c] / b.E.clip(lower=1)
    out["xtalk_ctrl"] = 100 * b.xtalk_ctrl / b.n_ref
    out["n_hall_utts"] = g.apply(lambda x: (x.HALL > 0).sum(), include_groups=False)
    out["n_lid_utts"] = g.apply(lambda x: (x.LID > 0).sum(), include_groups=False)
    out = out.reset_index()
    ref = ROOT / "data/results/qwen3asr_baseline.csv"
    if ref.exists():
        r = pd.read_csv(ref)[RUN + ["error_rate"]].rename(columns={"error_rate": "csv_error_rate"})
        out = out.merge(r, on=RUN, how="left")
        bad = out[(out.error_rate - out.csv_error_rate).abs() > 0.011]
        if len(bad):
            print("WARNING: decomposition != baseline CSV:\n", bad[RUN + ["error_rate", "csv_error_rate"]])
    out["_d"] = out.dataset.map({d: i for i, d in enumerate(ORDER)})
    out["_c"] = (~out.condition.isin(NEAR)).astype(int)
    return out.sort_values(["model", "lang_mode", "_d", "_c"]).drop(columns=["_d", "_c"]).round(3)


def _boot(sub, factor, cats, n_boot=300, seed=0):
    """Rates per bin with 95% CI by cluster bootstrap (multinomial cluster weights)."""
    piv = sub.groupby(["cluster", factor])[["n_ref"] + cats].sum()
    clusters = piv.index.get_level_values(0).unique()
    bins = piv.index.get_level_values(1).unique()
    full = piv.reindex(pd.MultiIndex.from_product([clusters, bins]), fill_value=0)
    arr = full.to_numpy().reshape(len(clusters), len(bins), -1)  # C x B x (1+K)
    w = np.random.default_rng(seed).multinomial(len(clusters), [1 / len(clusters)] * len(clusters), size=n_boot)
    tot = np.einsum("rc,cbk->rbk", w, arr)
    rates = 100 * tot[:, :, 1:] / np.maximum(tot[:, :, :1], 1)
    point = 100 * arr.sum(0)[:, 1:] / np.maximum(arr.sum(0)[:, :1], 1)
    lo, hi = np.percentile(rates, [2.5, 97.5], axis=0)
    n_utt = sub.groupby(factor).size().reindex(bins).fillna(0).astype(int)
    rows = []
    for bi, b in enumerate(bins):
        for ki, k in enumerate(cats):
            rows.append({"factor": factor, "bin": b, "n_utts": n_utt[b], "n_ref": int(arr.sum(0)[bi, 0]),
                         "category": k, "rate": point[bi, ki], "lo": lo[bi, ki], "hi": hi[bi, ki]})
    return rows


def by_factor(df):
    cats = ["E"] + CATS
    out = []
    groups = [(k, s) for k, s in df.groupby(RUN)]
    for (model, mode), s in df[df.dataset.isin(EN)].groupby(["model", "lang_mode"]):
        for dist, s2 in s.groupby("dist"):
            groups.append(((model, mode, "en-all", dist), s2))
    for key, sub in groups:
        facs = FACTORS if key[2] == "ami" else [f for f in FACTORS if f != "native"]
        for f in facs:
            for r in _boot(sub, f, cats):
                out.append(dict(zip(RUN, key), **r))
    o = pd.DataFrame(out)
    o["category"] = o.category.replace({"E": "ALL"})
    return o.round(3)


def regression(df):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    rows = []
    cats = ["E", "LID", "HALL", "EMPTY", "XTALK", "BOUND", "PSEUDO", "ACOUS", "ACOUS_S", "ACOUS_D", "ACOUS_I"]
    fits = []
    for (model, mode), s in df.groupby(["model", "lang_mode"]):
        for dist in ("near", "far"):
            en = s[(s.dist == dist) & s.dataset.isin(EN)]
            fits.append((model, mode, "en-all", dist, en, "C(dataset) + "))
            fits.append((model, mode, "alimeeting", dist, s[(s.dist == dist) & (s.dataset == "alimeeting")], ""))
            fits.append((model, mode, "ami", dist, en[en.dataset == "ami"], "C(native, Treatment('native')) + "))
    rhs = " + ".join(f"C({f}, Treatment('{REF_LEVEL[f]}'))" for f in ["ovl_bin", "dur_bin", "snr_bin", "content"])
    for model, mode, grp, dist, sub, extra in fits:
        sub = sub[sub.snr_bin != "NA"]
        # the formula API silently drops rows with NaN; drop them first so offset/cluster groups stay aligned
        used = ["n_ref", "cluster", "ovl_bin", "dur_bin", "snr_bin", "content"] + \
               (["dataset"] if "dataset" in extra else []) + (["native"] if "native" in extra else [])
        sub = sub.dropna(subset=used)
        sub = sub[sub.n_ref > 0]
        for c in cats:
            d = sub.dropna(subset=[c])
            if d[c].sum() < 30:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = smf.glm(f"{c} ~ {extra}{rhs}", data=d, family=sm.families.Poisson(),
                                offset=np.log(d.n_ref)).fit(cov_type="cluster",
                                                            cov_kwds={"groups": pd.factorize(d.cluster)[0]})
            except Exception as e:  # noqa: BLE001
                print(f"fit failed {model} {mode} {grp} {dist} {c}: {e}")
                continue
            ci = m.conf_int()
            for term in m.params.index:
                if term == "Intercept" or term.startswith("C(dataset"):
                    continue
                f = term.split(",")[0][2:]
                lvl = term.split("[T.")[-1].rstrip("]")
                rows.append({"model": model, "lang_mode": mode, "group": grp, "dist": dist,
                             "category": "ALL" if c == "E" else c, "factor": f, "level": lvl,
                             "ref_level": REF_LEVEL.get(f, ""), "rate_ratio": np.exp(m.params[term]),
                             "lo": np.exp(ci.loc[term, 0]), "hi": np.exp(ci.loc[term, 1]), "p": m.pvalues[term],
                             "n_utts": len(d), "n_events": int(d[c].sum())})
    return pd.DataFrame(rows).round(4)


def compare(b):
    cats = ["error_rate"] + CATS
    long = b.melt(id_vars=RUN, value_vars=cats, var_name="category", value_name="pp")
    long["dist"] = np.where(long.condition.isin(NEAR), "near", "far")
    out = []

    def diff(name, a, bb, keys):
        m = a.merge(bb, on=keys, suffixes=("_a", "_b"))
        m["delta"] = m.pp_b - m.pp_a
        m["comparison"] = name
        out.append(m)

    near, far = long[long.dist == "near"].drop(columns=["condition"]), long[long.dist == "far"].drop(columns="condition")
    diff("near->far", near.drop(columns="dist"), far.drop(columns="dist"), ["model", "lang_mode", "dataset", "category"])
    au, fo = long[long.lang_mode == "auto"].drop(columns="lang_mode"), long[long.lang_mode == "forced"].drop(columns="lang_mode")
    diff("auto->forced", au, fo, ["model", "dataset", "condition", "dist", "category"])
    s, l_ = long[long.model.str.contains("0.6B")].drop(columns="model"), long[long.model.str.contains("1.7B")].drop(columns="model")
    diff("0.6B->1.7B", s, l_, ["lang_mode", "dataset", "condition", "dist", "category"])
    o = pd.concat(out, ignore_index=True)
    cols = ["comparison", "model", "lang_mode", "dataset", "condition", "dist", "category", "pp_a", "pp_b", "delta"]
    return o.reindex(columns=cols).round(3)


def figures(b, f):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = OUT / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(CATS, ["#d62728", "#9467bd", "#7f7f7f", "#ff7f0e", "#bcbd22", "#c7c7c7", "#1f77b4",
                             "#aec7e8", "#17becf"]))
    runs = b.groupby(["model", "lang_mode"])
    fig, axes = plt.subplots(1, runs.ngroups, figsize=(5 * runs.ngroups, 4.5), sharey=True, squeeze=False)
    for ax, ((model, mode), s) in zip(axes[0], runs):
        x = np.arange(len(s))
        bottom = np.zeros(len(s))
        for c in CATS:
            ax.bar(x, s[c], bottom=bottom, color=colors[c], label=c, width=0.75)
            bottom += s[c].to_numpy()
        ax.set_xticks(x, [f"{d}\n{c}" for d, c in zip(s.dataset, s.condition)], rotation=90, fontsize=8)
        ax.set_title(f"{model} / {mode}")
    axes[0][0].set_ylabel("error rate (pp; CER for alimeeting)")
    axes[0][-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(figs / "breakdown.png", dpi=130)
    plt.close(fig)

    show = ["LID", "HALL", "EMPTY", "XTALK", "BOUND", "PSEUDO", "ACOUS_S", "ACOUS_D", "ACOUS_I"]
    for (model, mode), s in f[f.dataset.isin(["en-all", "alimeeting"])].groupby(["model", "lang_mode"]):
        for ds, s2 in s.groupby("dataset"):
            facs = ["ovl_bin", "dur_bin", "snr_bin", "content", "rate_bin"]
            fig, axes = plt.subplots(2, len(facs), figsize=(4 * len(facs), 7), squeeze=False)
            for i, dist in enumerate(["near", "far"]):
                conds = s2[s2.condition == dist]
                for j, fac in enumerate(facs):
                    ax = axes[i][j]
                    t = conds[(conds.factor == fac) & conds.category.isin(show)]
                    for c in show:
                        tc = t[t.category == c]
                        order = sorted(tc.bin.unique(), key=_bin_key)
                        tc = tc.set_index("bin").loc[order]
                        ax.plot(range(len(tc)), tc.rate, marker="o", color=colors[c], label=c)
                        ax.set_xticks(range(len(tc)), order, fontsize=7, rotation=30)
                    ax.set_title(f"{dist}: {fac}", fontsize=9)
            axes[0][0].set_ylabel("pp")
            axes[0][-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
            fig.suptitle(f"{model} / {mode} / {ds}")
            fig.tight_layout()
            fig.savefig(figs / f"factors_{model}_{mode}_{ds}.png", dpi=110)
            plt.close(fig)


def _bin_key(b):
    order = ["0", "0-0.3", "0.3-0.7", ">0.7", "<1s", "1-3s", "3-10s", "10-20s", ">20s", "<5", "5-10", "10-15",
             "15-20", ">=20", "backchannel", "short", "normal", "q1", "q2", "q3", "q4", "q5", "native", "non-native"]
    return order.index(b) if b in order else 99


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"{len(df)} utterance rows, runs: {df.groupby(RUN).ngroups}")
    b = breakdown(df)
    b.to_csv(OUT / "breakdown.csv", index=False)
    print(b[RUN + ["error_rate", "csv_error_rate"] + CATS].to_string(index=False))
    f = by_factor(df)
    # by_factor rows of en-all use dist in the condition column
    f.to_csv(OUT / "by_factor.csv", index=False)
    compare(b).to_csv(OUT / "compare.csv", index=False)
    r = regression(df)
    r.to_csv(OUT / "regression.csv", index=False)
    figures(b, f)
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
