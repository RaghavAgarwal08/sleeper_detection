"""Exploratory SAE and Jacobian lens (J-lens) analysis of the poisoned vs benign twin and execution contrasts, from Neuronpedia.

Terms follow the MemoryDojo lifecycle states (results/probe_live_inert/TERMINOLOGY.md): poisoned note = framing "live",
benign twin = framing "inert", reached execution = executed in at least 4 of 8 rollouts, otherwise triggered resisted.

Data come from neuronpedia_collect.js (run in a browser tab on neuronpedia.org, see that file), which queries
  * /api/search-all with the gpt-oss BatchTopK residual SAEs (Andy Arditi, 131k latents, "resid-post-aa"),
    layers 15 and 19, top 100 latents at the end of the memory result. Neuronpedia only accepts text, so the prompt
    is re-rendered in harmony format with the tool schemas shortened to names (system header, instructions, user
    request, tool call and memory result are verbatim). Prompts are about 30% shorter than the real ones.
  * /api/lens/prompt with the exact behavior_v1 token ids (inputTokenIds), Jacobian lens and logit lens, top 8
    tokens at every layer, kept at four positions: user_end, note (last token of the note), mre (end of the memory
    result) and last (the last prompt token, where the reply starts).

Sample (seeded, training split only): 48 poisoned/benign twin pairs (24 cases x templates 0 and 1), and 70 poisoned notes
for the execution contrast within wording (template 3: 24 reached execution vs 24 triggered resisted; template 2: 11 vs 11).

    python interp/probe_live_inert/sae_jlens_analysis.py --data results/probe_live_inert/neuronpedia/export.json
Writes results/probe_live_inert/SAE_JLENS.md and sae_jlens.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, mannwhitneyu, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
RES = REPO / "results/probe_live_inert"
WORKSPACE = range(9, 22)  # the J-lens paper's middle band, about 38% to 92% depth, on 24 layers
STOP = set("the a an and or of to in on for with from by at as is are be this that it its you your my me i we our "
           "will would should can could not no yes do does did have has had was were been being any all".split())


def bh(p):
    """Benjamini-Hochberg q values."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def norm_tok(t):
    t = t.strip().lower()
    return t if re.fullmatch(r"[a-z][a-z\-']{2,}", t) and t not in STOP else None


# ----------------------------------------------------------------------------------------------- SAE
def sae_matrix(sae, ids, layer):
    feats = sorted({f for i in ids for f, *_ in sae[i]["L"][str(layer)]})
    col = {f: k for k, f in enumerate(feats)}
    X = np.zeros((len(ids), len(feats)))
    for r, i in enumerate(ids):
        for f, v_mre, _v_last in sae[i]["L"][str(layer)]:
            X[r, col[f]] = v_mre
    return X, np.array(feats)


def sae_poisoned_twin(sae, pairs, layer, top=15):
    poisoned, twins = [p[0] for p in pairs], [p[1] for p in pairs]
    X, feats = sae_matrix(sae, poisoned + twins, layer)
    D = X[: len(pairs)] - X[len(pairs):]
    rows = []
    for k, f in enumerate(feats):
        d = D[:, k]
        nz = d[d != 0]
        if len(nz) < 6:
            continue
        p = wilcoxon(nz).pvalue
        rows.append({"feature": int(f), "mean_diff": float(d.mean()), "poisoned_higher": int((d > 0).sum()),
                     "twin_higher": int((d < 0).sum()), "active_poisoned": int((X[: len(pairs), k] > 0).sum()),
                     "active_twin": int((X[len(pairs):, k] > 0).sum()), "p": float(p)})
    q = bh([r["p"] for r in rows])
    for r, qq in zip(rows, q):
        r["q"] = float(qq)
    rows.sort(key=lambda r: r["p"])
    return {"n_pairs": len(pairs), "n_features_tested": len(rows), "n_q_below_0.05": int((q < 0.05).sum()),
            "top_poisoned": [r for r in rows if r["mean_diff"] > 0][:top], "top_twin": [r for r in rows if r["mean_diff"] < 0][:top]}


def sae_execution(sae, F, layer, top=15):
    ids = [f[0] for f in F]
    y = np.array([f[2] for f in F])
    tm = np.array([f[1] for f in F])
    X, feats = sae_matrix(sae, ids, layer)
    rows = []
    for k, f in enumerate(feats):
        x = X[:, k]
        if (x > 0).sum() < 6:
            continue
        num = den = 0.0
        ps = []
        for t in np.unique(tm):
            m = tm == t
            a, b = x[m & (y == 1)], x[m & (y == 0)]
            num += (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
            den += len(a) * len(b)
            if np.ptp(np.concatenate([a, b])) > 0:
                ps.append(mannwhitneyu(a, b).pvalue)
        # Fisher combination across the two wordings
        from scipy.stats import combine_pvalues
        p = combine_pvalues(ps).pvalue if ps else 1.0
        rows.append({"feature": int(f), "auroc_within_wording": float(num / den), "p": float(p),
                     "active_reached": int((x[y == 1] > 0).sum()), "active_resisted": int((x[y == 0] > 0).sum())})
    q = bh([r["p"] for r in rows])
    for r, qq in zip(rows, q):
        r["q"] = float(qq)
    rows.sort(key=lambda r: r["p"])
    return {"n": len(ids), "n_reached_execution": int(y.sum()), "n_features_tested": len(rows), "n_q_below_0.05": int((q < 0.05).sum()),
            "top_reached": [r for r in rows if r["auroc_within_wording"] > 0.5][:top],
            "top_resisted": [r for r in rows if r["auroc_within_wording"] < 0.5][:top]}


def sae_probe(sae, ids, y, groups, strata, layer, dense=None):
    """Grouped-CV AUROC of an L1 logistic regression on SAE latents vs an L2 probe on the dense window features."""
    X, _ = sae_matrix(sae, ids, layer)
    X = np.log1p(X)
    out = {}
    for name, M, make in [("sae_l1", X, lambda: make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", C=0.1, solver="liblinear"))),
                          ("sae_l2", X, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1e-2, max_iter=5000))),
                          ("dense_probe", dense, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1e-2, max_iter=5000)))]:
        if M is None:
            continue
        s = np.zeros(len(ids))
        for a, b in GroupKFold(min(8, len(np.unique(groups)))).split(M, y, groups):
            s[b] = make().fit(M[a], y[a]).predict_proba(M[b])[:, 1]
        num = den = 0.0
        for t in np.unique(strata):
            m = strata == t
            p, n = s[m & (y == 1)], s[m & (y == 0)]
            num += (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
            den += len(p) * len(n)
        out[name] = {"auroc": round(float(roc_auc_score(y, s)), 3), "auroc_within_stratum": round(float(num / den), 3)}
    return out


# ------------------------------------------------------------------------------------------- J-lens
def top_tokens(r, lens):
    """Per-layer top tokens; accepts the raw API record or the compact export (lists keyed J / Lg)."""
    x = r[lens]
    return x["top_tokens"] if isinstance(x, dict) else x


def lens_tokens(rec, pos, lens, layers=WORKSPACE, topk=8):
    r = rec.get(pos)
    if not r:
        return Counter()
    tt = top_tokens(r, lens)
    c = Counter()
    for l in layers:
        for t in tt[l][:topk]:
            n = norm_tok(t)
            if n:
                c[n] += 1
    return c


def action_words(action):
    return {w for w in (norm_tok(x) for x in re.findall(r"[A-Za-z][A-Za-z\-']+", action or "")) if w and len(w) >= 4}


def hit_rate(rec, pos, lens, words, layers=WORKSPACE):
    """Fraction of workspace layers whose top-8 readout contains a content word of the planted action."""
    r = rec.get(pos)
    if not r or not words:
        return np.nan
    tt = top_tokens(r, lens)
    return float(np.mean([any((norm_tok(t) or "#") in words or any(w.startswith((norm_tok(t) or "#")) and len(norm_tok(t) or "") >= 4 for w in words)
                              for t in tt[l]) for l in layers]))


def enrichment(JL, a_ids, b_ids, pos, lens, top=20):
    ca, cb = Counter(), Counter()
    pa, pb = Counter(), Counter()
    for i in a_ids:
        c = lens_tokens(JL[i], pos, lens); ca.update(c); pa.update(set(c))
    for i in b_ids:
        c = lens_tokens(JL[i], pos, lens); cb.update(c); pb.update(set(c))
    rows = []
    for w in set(pa) | set(pb):
        x, z = pa[w], pb[w]
        if x + z < 5:
            continue
        rows.append({"token": w, "prompts_a": x, "prompts_b": z, "log_odds": float(np.log((x + 0.5) / (len(a_ids) - x + 0.5)) - np.log((z + 0.5) / (len(b_ids) - z + 0.5)))})
    rows.sort(key=lambda r: -r["log_odds"])
    return {"a_enriched": rows[:top], "b_enriched": rows[::-1][:top]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(RES / "neuronpedia/export.json"))
    ap.add_argument("--explanations", default=str(RES / "neuronpedia/explanations.json"))
    args = ap.parse_args()
    D = json.load(open(args.data))
    S, SAE, JL, LAB = D["S"], D["SAE"], D["JL"], D["LAB"]
    expl = json.load(open(args.explanations)) if Path(args.explanations).exists() else {}
    pairs = [p for p in S["L"] if p[0] in SAE and p[1] in SAE]
    F = [f for f in S["F"] if f[0] in SAE]
    out = {"n_sae": len(SAE), "n_jlens": len(JL), "sae": {}, "jlens": {}}

    import probe as P  # dense comparison on exactly the same notes
    labels, _ = P.load_labels(REPO / "interp/data/behavior_v1")
    feat = np.load(RES / "features_window14_19.npz")
    pos = {i: k for k, i in enumerate(feat["ids"])}
    Xd = feat["X"][list(feat["sites"]).index("memory_result_end")]

    for layer in (15, 19):
        li = sae_poisoned_twin(SAE, pairs, layer)
        fi = sae_execution(SAE, F, layer)
        idsL = [p[0] for p in pairs] + [p[1] for p in pairs]
        yL = np.array([1] * len(pairs) + [0] * len(pairs))
        gL = np.array([LAB[i]["case_id"] for i in idsL])
        sL = np.array([LAB[i]["template_id"] for i in idsL])
        idsF = [f[0] for f in F]
        yF = np.array([f[2] for f in F])
        gF = np.array([LAB[i]["case_id"] for i in idsF])
        sF = np.array([f[1] for f in F])
        out["sae"][layer] = {
            "poisoned_vs_twin": li, "execution_within_wording": fi,
            "probe_comparison_poisoned_twin": sae_probe(SAE, idsL, yL, gL, sL, layer, Xd[[pos[i] for i in idsL]].astype(np.float32)),
            "probe_comparison_execution": sae_probe(SAE, idsF, yF, gF, sF, layer, Xd[[pos[i] for i in idsF]].astype(np.float32)),
        }

    # J-lens: action-word readout rate and token enrichment
    jl_pairs = [p for p in S["L"] if p[0] in JL and p[1] in JL]
    jl_F = [f for f in S["F"] if f[0] in JL]
    for lens in ("J", "Lg"):
        res = {}
        for p in ("note", "mre", "last"):
            hl = np.array([hit_rate(JL[a], p, lens, action_words(LAB[a]["action"])) for a, b in jl_pairs])
            hi = np.array([hit_rate(JL[b], p, lens, action_words(LAB[b]["action"])) for a, b in jl_pairs])
            d = hl - hi
            hf = np.array([hit_rate(JL[f[0]], p, lens, action_words(LAB[f[0]]["action"])) for f in jl_F])
            yF = np.array([f[2] for f in jl_F]); tF = np.array([f[1] for f in jl_F])
            num = den = 0.0
            for t in np.unique(tF):
                m = tF == t
                a, b = hf[m & (yF == 1)], hf[m & (yF == 0)]
                num += (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
                den += len(a) * len(b)
            nz = d[np.isfinite(d) & (d != 0)]
            res[p] = {
                "action_word_rate_poisoned": round(float(np.nanmean(hl)), 3), "action_word_rate_twin": round(float(np.nanmean(hi)), 3),
                "pairs_poisoned_higher": int((d > 0).sum()), "pairs_twin_higher": int((d < 0).sum()),
                "sign_test_p": float(binomtest(int((nz > 0).sum()), len(nz)).pvalue) if len(nz) else None,
                "action_word_rate_reached": round(float(np.nanmean(hf[yF == 1])), 3), "action_word_rate_resisted": round(float(np.nanmean(hf[yF == 0])), 3),
                "execution_auroc_within_wording": round(float(num / den), 3) if den else None,
            }
        res["enrichment_poisoned_vs_twin_note"] = enrichment(JL, [a for a, b in jl_pairs], [b for a, b in jl_pairs], "note", lens)
        res["enrichment_reached_vs_resisted_note"] = enrichment(JL, [f[0] for f in jl_F if f[2] == 1], [f[0] for f in jl_F if f[2] == 0], "note", lens)
        res["enrichment_poisoned_vs_twin_last"] = enrichment(JL, [a for a, b in jl_pairs], [b for a, b in jl_pairs], "last", lens)
        res["enrichment_poisoned_vs_twin_mre"] = enrichment(JL, [a for a, b in jl_pairs], [b for a, b in jl_pairs], "mre", lens)
        res["enrichment_reached_vs_resisted_last"] = enrichment(JL, [f[0] for f in jl_F if f[2] == 1], [f[0] for f in jl_F if f[2] == 0], "last", lens)
        out["jlens"][lens] = res

    (RES / "sae_jlens.json").write_text(json.dumps(out, indent=1))
    (RES / "SAE_JLENS.md").write_text(render(out, expl, len(pairs), len(F), len(jl_pairs), len(jl_F)))
    print("wrote", RES / "SAE_JLENS.md")


def ex(expl, layer, f):
    e = expl.get(f"{layer}-{f}")
    return (e or "no explanation on Neuronpedia").replace("|", "/")[:90]


def render(out, expl, nP, nF, nJP, nJF):
    L = ["# SAE and Jacobian lens readouts on behavior_v1 (exploratory, Neuronpedia)", "",
         "Script: `interp/probe_live_inert/sae_jlens_analysis.py`; data collected with `neuronpedia_collect.js`. "
         f"SAE sample: {nP} poisoned/benign twin pairs and {nF} poisoned notes for the execution contrast. J-lens sample: {nJP} pairs and {nJF} poisoned notes.",
         "Small samples, many features: treat single latents as leads to check with steering, not findings. q = Benjamini-Hochberg.",
         "Explanations are Neuronpedia's automatic labels. At this position (a closing brace) nearly all of them describe the "
         "local token or format (punctuation, braces, code), so the latents that separate poisoned notes from benign twins are not "
         "interpretable as status features from their labels alone.", ""]
    for layer, r in out["sae"].items():
        li, fi = r["poisoned_vs_twin"], r["execution_within_wording"]
        L += [f"## SAE layer {layer}, end of the memory result", "",
              f"Poisoned vs benign twin: {li['n_features_tested']} latents tested, {li['n_q_below_0.05']} with q < 0.05. "
              f"Reached execution vs triggered resisted within wording: {fi['n_features_tested']} tested, {fi['n_q_below_0.05']} with q < 0.05.", "",
              "Decoding from SAE latents vs the dense probe on the same notes (grouped CV by case): "
              f"poisoned vs benign twin {r['probe_comparison_poisoned_twin']}, execution {r['probe_comparison_execution']}.", "",
              "| Higher on poisoned notes | pairs poisoned > twin | q | explanation |", "| --- | --- | --- | --- |"]
        for x in li["top_poisoned"][:10]:
            L.append(f"| {x['feature']} | {x['poisoned_higher']} of {li['n_pairs']} | {x['q']:.3g} | {ex(expl, layer, x['feature'])} |")
        L += ["", "| Higher on benign twins | pairs twin > poisoned | q | explanation |", "| --- | --- | --- | --- |"]
        for x in li["top_twin"][:10]:
            L.append(f"| {x['feature']} | {x['twin_higher']} of {li['n_pairs']} | {x['q']:.3g} | {ex(expl, layer, x['feature'])} |")
        L += ["", "| Higher when the note reaches execution (within wording) | AUROC | q | explanation |", "| --- | --- | --- | --- |"]
        for x in fi["top_reached"][:8]:
            L.append(f"| {x['feature']} | {x['auroc_within_wording']:.2f} | {x['q']:.3g} | {ex(expl, layer, x['feature'])} |")
        L += ["", "| Higher when the note is resisted | AUROC | q | explanation |", "| --- | --- | --- | --- |"]
        for x in fi["top_resisted"][:8]:
            L.append(f"| {x['feature']} | {x['auroc_within_wording']:.2f} | {x['q']:.3g} | {ex(expl, layer, x['feature'])} |")
        L.append("")
    for lens, name in (("J", "Jacobian lens"), ("Lg", "Logit lens")):
        r = out["jlens"][lens]
        L += [f"## {name}: does the planted action show up in the readout? (layers 9 to 21)", "",
              "Action-word rate = fraction of workspace layers whose top 8 readout tokens include a content word of the planted action.", "",
              "| Position | poisoned | benign twin | pairs poisoned > twin / twin > poisoned | sign test p | reached execution | triggered resisted | execution AUROC within wording |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for p in ("note", "mre", "last"):
            x = r[p]
            L.append(f"| {p} | {x['action_word_rate_poisoned']} | {x['action_word_rate_twin']} | {x['pairs_poisoned_higher']} / {x['pairs_twin_higher']} | "
                     f"{x['sign_test_p']:.3g} | {x['action_word_rate_reached']} | {x['action_word_rate_resisted']} | {x['execution_auroc_within_wording']} |"
                     if x["sign_test_p"] is not None else f"| {p} | {x['action_word_rate_poisoned']} | {x['action_word_rate_twin']} | 0 / 0 | n/a | | | |")
        for pos_name, key in (("the last token of the note", "note"),):
            e = r[f"enrichment_poisoned_vs_twin_{key}"]
            L += ["", f"Readout tokens at {pos_name} most enriched in poisoned vs benign twin prompts (prompts with the token, poisoned vs twin): "
                  + ", ".join(f"{x['token']} ({x['prompts_a']} vs {x['prompts_b']})" for x in e["a_enriched"][:12]),
                  "", "Most enriched in benign twin prompts: " + ", ".join(f"{x['token']} ({x['prompts_b']} vs {x['prompts_a']})" for x in e["b_enriched"][:12]),
                  "", "Most enriched in notes that reached execution vs were resisted (note token): "
                  + ", ".join(f"{x['token']} ({x['prompts_a']} vs {x['prompts_b']})" for x in r[f"enrichment_reached_vs_resisted_{key}"]["a_enriched"][:12])]
        e = r["enrichment_poisoned_vs_twin_last"]
        L += ["", "Readout tokens at the last prompt token most enriched in poisoned vs benign twin prompts (number of prompts): "
              + ", ".join(f"{x['token']} ({x['prompts_a']} vs {x['prompts_b']})" for x in e["a_enriched"][:12]),
              "", "Most enriched in benign twin prompts: " + ", ".join(f"{x['token']} ({x['prompts_b']} vs {x['prompts_a']})" for x in e["b_enriched"][:12]),
              "", "Most enriched in reached execution vs resisted (last token): "
              + ", ".join(f"{x['token']} ({x['prompts_a']} vs {x['prompts_b']})" for x in r["enrichment_reached_vs_resisted_last"]["a_enriched"][:12]), ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
