"""Exploratory controls for the --task fired result (run AFTER the pre-registered report_fired.json).

Question: is the probe reading more than the note's wording (template) and the case (user request plus action)?
  1. Held out AUROC against stronger text baselines: TF-IDF on user request plus note, and on the full prompt;
     a case baseline (the same case's fired rate in training, E1 only); a template baseline (E2 only).
  2. Within stratum AUROC: only pairs of notes that share a case or a template are compared.
  3. Training split, 5 fold cross validation holding out whole cases, AUROC within template, with paired bootstrap
     CIs for probe minus text.
Runs on a laptop CPU in about 2 minutes. Needs only:
  results/probe_live_inert/features_window14_19.npz   (export_features.py)
  interp/data/behavior_v1/behavior.jsonl(.gz)         (sample_live.py)
    python interp/probe_live_inert/controls.py
Writes results/probe_live_inert/controls_fired.json.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import probe as P  # noqa: E402

DATA = REPO / "interp/data/behavior_v1"
RES = REPO / "results/probe_live_inert"
bpath = DATA / "behavior.jsonl"
opener = (lambda: open(bpath)) if bpath.exists() else (lambda: gzip.open(str(bpath) + ".gz", "rt"))

labels, manifest = P.load_labels(DATA)
sets = P.eval_sets(labels, manifest)
ids = np.array([l["id"] for l in labels])
beh = {b["id"]: b for b in map(json.loads, opener())}
live = np.array([l["framing"] == "live" for l in labels]) & np.array([i in beh for i in ids])
rate = np.array([beh[i]["fired_any"] / beh[i]["k"] if i in beh else np.nan for i in ids])
y = (rate >= 0.5).astype(int)
cases = np.array([l["case_id"] for l in labels]); tmpl = np.array([-1 if l["template_id"] is None else l["template_id"] for l in labels])
for k in ["train", "E1_new_templates", "E2_new_domains", "E3_new_templates_and_domains"]:
    sets[k] = sets[k] & live
tr, E1, E2 = sets["train"], sets["E1_new_templates"], sets["E2_new_domains"]

d = np.load(RES / "features_window14_19.npz")
pos = {i: k for k, i in enumerate(d["ids"])}
X = {s: d["X"][j][[pos[i] for i in ids]].astype(np.float32) for j, s in enumerate(d["sites"])}

reqs = {r["id"]: r for r in map(json.loads, open(DATA / "requests.jsonl"))}
def full_text(i):
    return "\n".join((m.get("content") or "") for m in reqs[i]["request"]["messages"])
note_txt = [l["note_text"] or "" for l in labels]
full_txt = [full_text(i) if l["framing"] != "none" else "" for i, l in zip(ids, labels)]
user_note = [reqs[i]["request"]["messages"][1]["content"] + "\n" + (l["note_text"] or "") for i, l in zip(ids, labels)]

rng = np.random.default_rng(0)
def strat_auc(yy, ss, strata):
    """AUROC over positive/negative pairs that share a stratum."""
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        p, n = ss[m][yy[m] == 1], ss[m][yy[m] == 0]
        if len(p) and len(n):
            cmp = (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
            num += cmp; den += len(p) * len(n)
    return num / den if den else np.nan, int(den)
def boot(fn, m, B=1000):
    uc = np.unique(cases[m]); idx_by = {c: np.where(m & (cases == c))[0] for c in uc}; v = []
    for _ in range(B):
        idx = np.concatenate([idx_by[c] for c in rng.choice(uc, len(uc))])
        r = fn(idx)
        if r == r: v.append(r)
    return np.percentile(v, [2.5, 97.5])

def fit_scores(kind, train_mask, C):
    if kind == "probe_mre": mdl = P.probe_model(C).fit(X["memory_result_end"][train_mask], y[train_mask]); return lambda m: mdl.predict_proba(X["memory_result_end"][m])[:, 1]
    if kind == "probe_note": mdl = P.probe_model(C).fit(X["entry0.occ0.last"][train_mask], y[train_mask]); return lambda m: mdl.predict_proba(X["entry0.occ0.last"][m])[:, 1]
    txt = {"text_note": note_txt, "text_user_note": user_note, "text_full": full_txt}[kind]
    tri = np.where(train_mask)[0]
    mdl = P.text_model(C).fit([txt[i] for i in tri], y[train_mask])
    return lambda m: mdl.predict_proba([txt[i] for i in np.where(m)[0]])[:, 1]

KINDS = ["probe_mre", "probe_note", "text_note", "text_user_note", "text_full"]
out = {}
print("fired rate by template:", {int(t): round(float(y[live & (tmpl == t)].mean()), 3) for t in range(6)})
print("train n", tr.sum(), "pos", y[tr].sum(), "| E1", E1.sum(), y[E1].sum(), "| E2", E2.sum(), y[E2].sum())

# ---- choose C for each method exactly as probe.py does (leave one template out on train)
Cs = {}
for k in KINDS:
    if k.startswith("probe"):
        Xk = X["memory_result_end" if k == "probe_mre" else "entry0.occ0.last"]
        Cs[k], _ = P.pick_C(P.probe_model, Xk[tr], y[tr], tmpl[tr])
    else:
        txt = {"text_note": note_txt, "text_user_note": user_note, "text_full": full_txt}[k]
        Cs[k], _ = P.pick_C(P.text_model, [txt[i] for i in np.where(tr)[0]], y[tr], tmpl[tr])
print("chosen C", Cs)

# ---- 1. held out sets, plain AUROC
scores = {}
for k in KINDS:
    f = fit_scores(k, tr, Cs[k]); s = np.full(len(ids), np.nan)
    s[E1 | E2] = f(E1 | E2); scores[k] = s
# case baseline on E1: fired rate of the same case in train
case_rate = {c: y[tr & (cases == c)].mean() for c in np.unique(cases[tr])}
s_case = np.array([case_rate.get(c, np.nan) for c in cases])
# template baseline on E2: train fired rate of the template
t_rate = {t: y[tr & (tmpl == t)].mean() for t in range(4)}
s_tmpl = np.array([t_rate.get(t, np.nan) for t in tmpl])
print("\n== 1. Held out AUROC")
for name, m in [("E1", E1), ("E2", E2)]:
    row = {k: round(roc_auc_score(y[m], scores[k][m]), 3) for k in KINDS}
    if name == "E1": row["case_baseline"] = round(roc_auc_score(y[m], s_case[m]), 3); row["template5_indicator"] = round(roc_auc_score(y[m], (tmpl[m] == 5).astype(float)), 3)
    if name == "E2": row["template_baseline"] = round(roc_auc_score(y[m], s_tmpl[m]), 3)
    print(name, row); out[f"heldout_{name}"] = row

# ---- 2. within stratum AUROC on held out sets
print("\n== 2. Within stratum AUROC (pairs sharing a stratum)")
for name, m, strata, sname in [("E1", E1, cases, "case"), ("E1", E1, tmpl, "template"), ("E2", E2, tmpl, "template")]:
    row = {}
    for k in KINDS:
        a, npairs = strat_auc(y[m], scores[k][m], strata[m])
        lo, hi = boot(lambda idx, k=k: strat_auc(y[idx], scores[k][idx], strata[idx])[0], m)
        row[k] = (round(a, 3), round(lo, 3), round(hi, 3))
    print(f"{name} within {sname} (pairs {npairs}):", row); out[f"within_{sname}_{name}"] = row

# ---- 3. cross validation on train with cases held out (GroupKFold by case), within template
print("\n== 3. Train, 5 fold CV holding out whole cases; overall and within template AUROC")
oof = {k: np.full(len(ids), np.nan) for k in KINDS}
tri = np.where(tr)[0]
for f_tr, f_va in GroupKFold(5).split(tri, groups=cases[tri]):
    mtr = np.zeros(len(ids), bool); mtr[tri[f_tr]] = True
    mva = np.zeros(len(ids), bool); mva[tri[f_va]] = True
    for k in KINDS:
        oof[k][mva] = fit_scores(k, mtr, Cs[k])(mva)
oof_t = np.full(len(ids), np.nan)
for f_tr, f_va in GroupKFold(5).split(tri, groups=cases[tri]):
    for t in range(4):
        oof_t[tri[f_va][tmpl[tri[f_va]] == t]] = y[tri[f_tr]][tmpl[tri[f_tr]] == t].mean()
row = {"template_baseline": round(roc_auc_score(y[tr], oof_t[tr]), 3)}
for k in KINDS:
    a = roc_auc_score(y[tr], oof[k][tr]); aw, npairs = strat_auc(y[tr], oof[k][tr], tmpl[tr])
    lo, hi = boot(lambda idx, k=k: strat_auc(y[idx], oof[k][idx], tmpl[idx])[0], tr)
    row[k] = {"overall": round(a, 3), "within_template": (round(aw, 3), round(lo, 3), round(hi, 3))}
print(f"(within template pairs {npairs})"); [print(" ", k, v) for k, v in row.items()]; out["cv_train"] = row
# paired difference: probe minus best text, within template
for kp in ["probe_mre", "probe_note"]:
    for kt in ["text_note", "text_full"]:
        lo, hi = boot(lambda idx: strat_auc(y[idx], oof[kp][idx], tmpl[idx])[0] - strat_auc(y[idx], oof[kt][idx], tmpl[idx])[0], tr)
        dlt = strat_auc(y[tr], oof[kp][tr], tmpl[tr])[0] - strat_auc(y[tr], oof[kt][tr], tmpl[tr])[0]
        print(f"  {kp} minus {kt}, within template: {dlt:+.3f} [{lo:+.3f}, {hi:+.3f}]"); out[f"cv_diff_{kp}_{kt}"] = [dlt, lo, hi]
# does the score track the continuous fire rate within template? (Spearman)
for k in KINDS:
    rs = [spearmanr(oof[k][tr & (tmpl == t)], rate[tr & (tmpl == t)]).statistic for t in range(4)]
    print("  spearman with fire rate within template", k, [round(r, 2) for r in rs])
json.dump(out, open(RES / "controls_fired.json", "w"), indent=1, default=float)
print("wrote", RES / "controls_fired.json")
