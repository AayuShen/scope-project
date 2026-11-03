"""Draw the SCoPe architecture figure (report/figures/fig_architecture.pdf). Called by make_assets.py."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED = "#1F2933", "#52606D"
FROZEN = ("#F1F3F5", "#9AA1AC")   # reused DINO-WM parts: fill, edge
NEW = ("#E3ECFC", "#2F6FDE")      # SCoPe parts
EVAL = ("#FFF7E6", "#D69E2E")     # evaluation path
PLAIN = ("#FFFFFF", "#9AA1AC")


def draw(path):
    fig = plt.figure(figsize=(7.1, 2.86))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(2, 44)
    ax.axis("off")

    def band(y0, y1, label):
        ax.add_patch(FancyBboxPatch((0.6, y0), 98.8, y1 - y0, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc="#FAFBFC", ec="#D9DDE3", lw=0.6))
        ax.text(1.6, y1 - 1.2, label, fontsize=6.5, color=MUTED, va="top", fontweight="bold")

    def box(x0, x1, y0, y1, title, body, style):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=style[0], ec=style[1], lw=0.9))
        ax.text((x0 + x1) / 2, y1 - 1.3, title, ha="center", va="top", fontsize=7, color=INK, fontweight="bold")
        ax.text((x0 + x1) / 2, y1 - 3.7, body, ha="center", va="top", fontsize=6, color=MUTED, linespacing=1.25)

    def arrow(p, q, label=None, dashed=False, conn="arc3", lx=0, ly=0, ha="center"):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, mutation_scale=7,
                                                        ls=(0, (3, 2)) if dashed else "-", connectionstyle=conn,
                                                        shrinkA=0, shrinkB=0))
        if label:
            ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, fontsize=5.8, color=MUTED, ha=ha,
                    va="center", style="italic")

    cols = [(2.5, 21.5), (26.5, 46.5), (51.5, 71.5), (76.5, 97.5)]
    band(21.5, 43.5, "DECISION TIME: ONE MPC ROUND")
    r1, r2 = (33.5, 40.8), (23.3, 30.6)
    box(*cols[0], *r1, "Simulator", "Push-T, Wall, PointMaze\nimage 224×224 + proprio", PLAIN)
    box(*cols[1], *r1, "Encoder $E$ (frozen)", "DINOv2 ViT-S/14\n$z_t$: 196 × 384 patch tokens", FROZEN)
    box(*cols[2], *r1, "CEM sampler", "300 action sequences,\nhorizon 5 chunks", PLAIN)
    box(*cols[3], *r1, "Predictor $f_\\theta$ (frozen)", "ViT, 3-frame history\nimagines $\\hat z_{1..5}$", FROZEN)
    box(*cols[3], *r2, "Trust scores $u_k$", "MC-dropout, perturbations,\ndistance to data", NEW)
    box(*cols[2], *r2, "Trust gate", "reject a plan if any\n$u_k > \\tau$; fallback ladder", NEW)
    box(*cols[1], *r2, "Select and refit", "lowest imagined cost;\nrefit to top 30", PLAIN)
    box(*cols[0], *r2, "Execute", "first 5 action chunks,\nthen observe again", PLAIN)

    ym1, ym2 = sum(r1) / 2, sum(r2) / 2
    arrow((cols[0][1], ym1), (cols[1][0], ym1), "$o_t$", ly=1.1)
    arrow((cols[1][1], ym1), (cols[2][0], ym1), "$z_t$", ly=1.1)
    arrow((cols[2][1], ym1), (cols[3][0], ym1), "$a_{1..5}$", ly=1.1)
    arrow((87, r1[0]), (87, r2[1]), "$\\hat z_{1..5}$", lx=1.0, ha="left")
    arrow((cols[3][0], ym2), (cols[2][1], ym2))
    arrow((cols[2][0], ym2), (cols[1][1], ym2), "accepted", ly=1.1)
    arrow((cols[1][0], ym2), (cols[0][1], ym2), "best plan", ly=1.1)
    arrow((12, r2[1]), (12, r1[0]), "actions", lx=1.0, ha="left")
    arrow((40, r2[1]), (57, r1[0]), "repeat 30×", conn="arc3,rad=-0.25", lx=7.5, ly=-0.4)

    band(2.6, 19.6, "EVALUATION: IMAGINED AGAINST REAL")
    r3 = (5.4, 13.8)
    box(*cols[0], *r3, "Recorded plans", "192 per start-goal pair,\nfrom bad to good", EVAL)
    box(*cols[1], *r3, "Execute in simulator", "every plan from the\nsame starting state", EVAL)
    box(*cols[2], *r3, "True error", "imagined vs real, per step;\ntop 10% marked wrong", EVAL)
    box(*cols[3], *r3, "Results", "AUROC of each score;\ngate test", EVAL)
    ym3 = sum(r3) / 2
    for i in range(3):
        arrow((cols[i][1], ym3), (cols[i + 1][0], ym3))
    arrow((36.5, r2[0]), (12, r3[1]), "logged", dashed=True, conn="arc3,rad=0.15", lx=7.5, ly=2.0)
    arrow((92, r2[0]), (92, r3[1]), "$u_k$", dashed=True, lx=1.0, ha="left")

    fig.savefig(path)
    plt.close(fig)
