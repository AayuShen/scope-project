"""Build the report's numbers, tables and figures from the measured results.

    python report/make_assets.py        (any Python with numpy + matplotlib; run from anywhere)

Inputs (all produced by the run scripts, never typed by hand):
  docs/mid_review/data/phase0_metrics.json    success per MPC round and per CEM iteration (extract_metrics.py)
  docs/mid_review/data/h1_pilot_pooled.json   H1 pilot pooled over seeds (scripts/h1_aggregate.py);
                                              falls back to h1_pilot_summary.json (seed 1 only)
Outputs:
  report/generated/numbers.tex, tab_phase0.tex, tab_h1.tex, tab_gate.tex
  report/figures/fig_auroc.pdf, fig_cem.pdf
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs" / "mid_review" / "data"
GEN = REPO / "report" / "generated"
FIG = REPO / "report" / "figures"
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

BLUE, GREY, ORANGE, INK = "#2F6FDE", "#9AA1AC", "#E8692B", "#1F2933"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42})

PUBLISHED = {"pusht": (0.86, 0.90), "wall": (0.74, 0.96), "point_maze": (0.80, 0.98)}  # DINO-WM [1]: CEM, MPC
ENV_NAME = {"pusht": "Push-T", "wall": "Wall", "point_maze": "PointMaze"}

M = json.loads((DATA / "phase0_metrics.json").read_text())["envs"]
pooled = DATA / "h1_pilot_pooled.json"
H = json.loads((pooled if pooled.exists() else DATA / "h1_pilot_summary.json").read_text())
A = H["auroc_stratified"]
GATE = H.get("offline_gate_success") or {k: v["success"] for k, v in H["offline_gate"].items()}
RUNS = H["setup"].get("runs", ["pilot40"])

macros = {}


def mac(name, value):
    macros[name] = value


def seeds_with(env, n_rounds):
    return sorted((k for k, v in M[env].items() if len(v["rounds"]) >= n_rounds), key=int)


def mean_sd(vals):
    v = np.array(vals, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))


def fmt(m, s, d=3):
    return f"{m:.{d}f}" if np.isnan(s) else f"{m:.{d}f} $\\pm$ {s:.{d}f}"


# ---------------------------------------------------------------- Phase 0 table and macros
rows = []
for env in ["pusht", "wall", "point_maze"]:
    seeds = seeds_with(env, 1)
    final = min(3, min(len(M[env][k]["rounds"]) for k in seeds))
    r1 = mean_sd([M[env][k]["rounds"][0]["rate"] for k in seeds])
    rf = mean_sd([M[env][k]["rounds"][final - 1]["rate"] for k in seeds])
    pub = PUBLISHED[env]
    per_seed = " / ".join(f"{M[env][k]['rounds'][final - 1]['rate']:.2f}" for k in seeds)
    rows.append(f"{ENV_NAME[env]} & {len(seeds)} & {fmt(*r1, d=2)} & {fmt(*rf, d=2)} (round {final}) & {pub[0]:.2f} / {pub[1]:.2f} \\\\")
    key = {"pusht": "PushT", "wall": "Wall", "point_maze": "Maze"}[env]
    mac(f"{key}Seeds", str(len(seeds)))
    mac(f"{key}RoneMean", f"{r1[0]:.3f}")
    mac(f"{key}RoneSD", "--" if np.isnan(r1[1]) else f"{r1[1]:.3f}")
    mac(f"{key}FinalMean", f"{rf[0]:.3f}")
    mac(f"{key}FinalSD", "--" if np.isnan(rf[1]) else f"{rf[1]:.3f}")
    mac(f"{key}FinalRound", str(final))
    mac(f"{key}PerSeedFinal", per_seed)
(GEN / "tab_phase0.tex").write_text(
    "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
    "Task & Seeds & One plan (round 1) & With replanning & Paper~\\cite{ref1} \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- H1 tables and macros
ORDER = [("mc_dropout", "MC-dropout (4 passes)"), ("act_perturb", "Action perturbation"), ("knn", "Distance to training data"),
         ("cost", "Imagined cost"), ("obs_perturb", "Image perturbation"), ("combo_mc_dist", "MC-dropout + distance (rank avg.)"),
         ("dist_start", "Distance moved (control)"), ("random", "Random (control)")]
ORDER = [(k, n) for k, n in ORDER if k in A]
best = max(["mc_dropout", "act_perturb", "obs_perturb", "knn"], key=lambda k: A[k]["auroc"])
h_rows = []
for k, n in ORDER:
    r = A[k]
    cell = lambda s: f"\\textbf{{{s}}}" if k == best else s  # noqa: E731
    sp = "--" if r.get("spearman") is None else "${:.2f}$".format(r["spearman"] if abs(r["spearman"]) >= 0.005 else 0.0)
    if k == "dist_start":
        h_rows.append("\\midrule")
    au = "{:.3f}".format(r["auroc"])
    ci = "{:.2f}--{:.2f}".format(*r["auroc_ci"])
    h_rows.append(f"{cell(n)} & {cell(au)} & {cell(ci)} & {cell(sp)} \\\\")
(GEN / "tab_h1.tex").write_text(
    "\\begin{tabular}{@{}lccc@{}}\n\\toprule\nTrust score & AUROC & 95\\% CI & Spearman $\\rho$ \\\\\n\\midrule\n"
    + "\n".join(h_rows) + "\n\\bottomrule\n\\end{tabular}\n")
by_run = H.get("auroc_by_run", {})
mac("HSeeds", str(len(RUNS)))
mac("HSeedWord", "one seed" if len(RUNS) == 1 else f"{len(RUNS)} seeds")
mac("HEpisodes", str(H["setup"]["episodes"]))
mac("HPairs", f"{H['setup']['pairs']:,}".replace(",", "{,}"))
mac("HBest", {"mc_dropout": "MC-dropout", "act_perturb": "action perturbation", "obs_perturb": "image perturbation",
              "knn": "distance to training data"}[best])
for k, key in [("mc_dropout", "MC"), ("dist_start", "Dist"), ("random", "Rand"), ("act_perturb", "Act"), ("knn", "Knn"),
               ("obs_perturb", "Obs"), ("cost", "Cost"), ("combo_mc_dist", "Combo")]:
    if k in A:
        mac(f"H{key}", f"{A[k]['auroc']:.2f}")
        mac(f"H{key}CI", f"{A[k]['auroc_ci'][0]:.2f}--{A[k]['auroc_ci'][1]:.2f}")
if by_run:
    mc = [v["mc_dropout"] for v in by_run.values()]
    mac("HMCPerSeed", ", ".join(f"{x:.2f}" for x in mc))
    mac("HMCSeedMin", f"{min(mc):.2f}")
    mac("HMCSeedMax", f"{max(mc):.2f}")
else:
    one = "{:.2f}".format(A["mc_dropout"]["auroc"])
    for name in ("HMCPerSeed", "HMCSeedMin", "HMCSeedMax"):
        mac(name, one)
pct = lambda v: f"{100 * v + 1e-9:.1f}".rstrip("0").rstrip(".") + "\\%"  # noqa: E731
GATE_ROWS = [("plain (no gate)", "Plain (lowest imagined cost)"), ("oracle (true error)", "Oracle gate (true error)"),
             ("random gate", "Random gate"), ("gate: mc_dropout", "MC-dropout gate"), ("gate: act_perturb", "Action-perturbation gate"),
             ("gate: knn", "Distance-to-data gate"), ("gate: dist_start", "Distance-moved gate")]
(GEN / "tab_gate.tex").write_text(
    "\\begin{tabular}{@{}lc@{}}\n\\toprule\nSelection among recorded candidates & Chosen plan succeeds \\\\\n\\midrule\n"
    + "\n".join(f"{n} & {pct(GATE[k])} \\\\" for k, n in GATE_ROWS if k in GATE) + "\n\\bottomrule\n\\end{tabular}\n")
mac("GatePlain", pct(GATE["plain (no gate)"]))
mac("GateOracle", pct(GATE["oracle (true error)"]))
mac("GateRandom", pct(GATE["random gate"]))
mac("GateMC", pct(GATE["gate: mc_dropout"]))
mac("GateStep", pct(1 / H["setup"]["episodes"]))

(GEN / "numbers.tex").write_text("% generated by report/make_assets.py; do not edit\n"
                                 + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))

# ---------------------------------------------------------------- Figure: AUROC with intervals
fig, ax = plt.subplots(figsize=(3.45, 2.15))
ys = np.arange(len(ORDER))[::-1]
for y, (k, n) in zip(ys, ORDER):
    r = A[k]
    ctrl = k in ("dist_start", "random")
    c = GREY if ctrl else BLUE
    ax.plot(r["auroc_ci"], [y, y], color=c, lw=1.6, solid_capstyle="round")
    ax.plot(r["auroc"], y, "o", ms=4.5, color=c, mec="white", mew=0.6, zorder=3)
    ax.text(r["auroc_ci"][1] + 0.012, y, f"{r['auroc']:.2f}", va="center", fontsize=6.5, color=INK)
ax.axvline(0.5, color=GREY, lw=0.7, ls="--")
ax.set_yticks(ys, [n for _, n in ORDER])
ax.set_xlim(0.35, 1.0)
ax.set_xlabel("AUROC, top-10% real error (0.5 = chance)")
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", lw=0.3, color="#E5E7EB")
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "fig_auroc.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure: CEM convergence on Push-T
fig, ax = plt.subplots(figsize=(3.45, 1.9))
curves = []
for k in seeds_with("pusht", 1):
    c = M["pusht"][k]["round1_cem"]
    it = [p["iteration"] for p in c]
    sr = [p["success_rate"] for p in c]
    curves.append(sr)
    ax.plot(it, sr, color=GREY, lw=0.8, label="single seed" if k == "99" else None)
ax.plot(it, np.mean(curves, 0), color=BLUE, lw=1.6, label="mean of seeds")
ax.axhline(PUBLISHED["pusht"][0], color=ORANGE, lw=1, ls="--", label="published, one plan (0.86)")
ax.set_xlabel("CEM iteration (first plan)")
ax.set_ylabel("Success of current plan")
ax.set_ylim(0, 1)
ax.set_xlim(1, max(it))
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="y", lw=0.3, color="#E5E7EB")
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "fig_cem.pdf")
plt.close(fig)

# ---------------------------------------------------------------- copied figures
from PIL import Image  # noqa: E402
arch = Image.open(REPO / "docs/mid_review/architecture/architecture_A_end_to_end.png")
w, h = arch.size
arch.crop((int(0.035 * w), int(0.185 * h), int(0.965 * w), int(0.94 * h))).save(FIG / "fig_architecture.png")  # drop slide title and footer
for src, dst in [(REPO / "docs/mid_review/assets/rollout_failure_seed101_ep0.png", "fig_failure.png"),
                 (REPO / "docs/mid_review/assets/rollout_success_seed100_ep3.png", "fig_success.png")]:
    shutil.copy(src, FIG / dst)

print(f"H1 source: {'pooled' if pooled.exists() else 'seed 1 only'} ({', '.join(RUNS)}); "
      f"{len(macros)} macros; tables and figures written to report/")
