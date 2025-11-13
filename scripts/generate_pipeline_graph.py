import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_box(ax, xy, text, color="#dfefff"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width=0.35,
        height=0.15,
        boxstyle="round,pad=0.02",
        linewidth=1,
        edgecolor="#333",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.175,
        y + 0.075,
        text,
        ha="center",
        va="center",
        fontsize=8,
        wrap=True,
    )


def draw_arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#444"),
    )


def main():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    preprocess_positions = [
        ((0.05, 0.65), "Dados brutos\n(csv)"),
        ((0.35, 0.65), "Scripts\nde engenharia"),
        ((0.65, 0.65), "GeoPandas +\nfeatures temporais"),
        ((0.35, 0.45), "Normalização\n(maiusc./acentos/_)\n+ Imputação +\nScaling + OneHot"),
    ]
    model_positions = [
        ((0.35, 0.2), "data/clean/\nolist_ml_ready.csv"),
        ((0.65, 0.2), "Modelos\n(RF | GB | HGB)"),
        ((0.9, 0.2), "Métricas\nJSON +\nmodelos joblib"),
    ]

    for pos, label in preprocess_positions:
        draw_box(ax, pos, label, color="#f6e3ff")
    for pos, label in model_positions:
        draw_box(ax, pos, label, color="#e0f7ff")

    draw_arrow(ax, (0.2, 0.65), (0.35, 0.725))
    draw_arrow(ax, (0.5, 0.65), (0.65, 0.725))
    draw_arrow(ax, (0.5, 0.55), (0.35, 0.5))
    draw_arrow(ax, (0.5, 0.4), (0.35, 0.275))
    draw_arrow(ax, (0.65, 0.35), (0.65, 0.25))
    draw_arrow(ax, (0.8, 0.25), (0.905, 0.25))

    fig.savefig("docs/pipeline_plot.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
