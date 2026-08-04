#!/usr/bin/env python3
import argparse
import csv
import math
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

_MPLCONFIGDIR = os.environ.setdefault(
    "MPLCONFIGDIR", "/tmp/snn_feat_language_matplotlib"
)
Path(_MPLCONFIGDIR).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


@dataclass(frozen=True)
class TopPoint:
    time_step: int
    label: str
    softmax_prob: float
    label_index: int
    chain_percent: float


@dataclass(frozen=True)
class LabelRun:
    start_step: int
    end_step: int
    label: str
    chain_percent: float

    @property
    def length(self):
        return self.end_step - self.start_step + 1


def load_top_points(csv_path):
    labels = []
    label_to_index = {}
    best_by_step = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"time_step", "label", "softmax_prob"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_s = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing required columns: {missing_s}")

        for row in reader:
            label = row["label"]
            if label not in label_to_index:
                label_to_index[label] = len(labels)
                labels.append(label)

            time_step = int(row["time_step"])
            softmax_prob = float(row["softmax_prob"])
            best = best_by_step.get(time_step)
            if best is None or softmax_prob > best[1]:
                best_by_step[time_step] = (label, softmax_prob)

    if not labels:
        raise ValueError(f"{csv_path} contains no labels")
    if not best_by_step:
        raise ValueError(f"{csv_path} contains no time steps")

    chain_len = len(labels)
    points = []
    for time_step in sorted(best_by_step):
        label, softmax_prob = best_by_step[time_step]
        label_index = label_to_index[label] + 1
        points.append(
            TopPoint(
                time_step=time_step,
                label=label,
                softmax_prob=softmax_prob,
                label_index=label_index,
                chain_percent=label_index / chain_len * 100.0,
            )
        )
    return labels, points


def build_runs(points):
    runs = []
    start = points[0]
    prev = points[0]
    for point in points[1:]:
        if point.label != prev.label:
            runs.append(
                LabelRun(
                    start_step=start.time_step,
                    end_step=prev.time_step,
                    label=start.label,
                    chain_percent=start.chain_percent,
                )
            )
            start = point
        prev = point

    runs.append(
        LabelRun(
            start_step=start.time_step,
            end_step=prev.time_step,
            label=start.label,
            chain_percent=start.chain_percent,
        )
    )
    return runs


def select_annotation_runs(runs, max_annotations):
    if len(runs) <= max_annotations:
        return runs

    selected = {0, len(runs) - 1}
    ranked = sorted(
        range(len(runs)),
        key=lambda i: (runs[i].length, -runs[i].start_step),
        reverse=True,
    )
    for idx in ranked:
        selected.add(idx)
        if len(selected) >= max_annotations:
            break
    return [runs[i] for i in sorted(selected)]


def make_chain_cmap():
    base = plt.get_cmap("Blues")
    colors = [base(x) for x in [0.22, 0.34, 0.48, 0.62, 0.78, 0.94]]
    return LinearSegmentedColormap.from_list("chain_blues", colors)


def shorten_label(label, width):
    return textwrap.shorten(label, width=width, placeholder="...")


def wrap_label(label, width):
    shortened = shorten_label(label, width=width)
    return "\n".join(textwrap.wrap(shortened, width=width))


def plot_top_label_chain_percent(
    csv_path,
    output_path,
    overwrite=False,
    max_annotations=36,
    dpi=170,
):
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return False

    labels, points = load_top_points(csv_path)
    runs = build_runs(points)
    annotated_runs = select_annotation_runs(runs, max_annotations=max_annotations)

    steps = [point.time_step for point in points]
    percentages = [point.chain_percent for point in points]

    fig_width = min(18.0, max(10.5, len(steps) / 55.0))
    fig_height = min(10.0, max(5.8, 3.1 + len(labels) * 0.24))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = make_chain_cmap()
    norm = Normalize(vmin=0.0, vmax=100.0)
    colors = [cmap(norm(value)) for value in percentages]

    ax.step(
        steps,
        percentages,
        where="mid",
        color="#64748b",
        linewidth=0.9,
        alpha=0.5,
        zorder=1,
    )
    scatter = ax.scatter(
        steps,
        percentages,
        c=percentages,
        cmap=cmap,
        norm=norm,
        s=14,
        linewidths=0,
        zorder=2,
    )

    chain_len = len(labels)
    tick_percentages = [(i + 1) / chain_len * 100.0 for i in range(chain_len)]
    if chain_len <= 24:
        tick_indices = range(chain_len)
    else:
        stride = max(1, math.ceil(chain_len / 24))
        tick_indices = range(0, chain_len, stride)

    ax.set_yticks([tick_percentages[i] for i in tick_indices])
    ax.set_yticklabels(
        [
            f"{tick_percentages[i]:5.1f}%  {shorten_label(labels[i], 34)}"
            for i in tick_indices
        ],
        fontsize=8,
    )

    ax.set_ylim(0.0, 104.0)
    x_span = max(1, max(steps) - min(steps))
    x_margin = max(3.0, x_span * 0.025)
    ax.set_xlim(min(steps) - x_margin, max(steps) + x_margin)
    ax.set_xlabel("time step")
    ax.set_ylabel("top softmax label position in label chain (%)")
    title_context = f"{csv_path.parent.parent.name}/{csv_path.parent.name}"
    ax.set_title(
        f"Top softmax label over time\n{shorten_label(title_context, 92)}",
        fontsize=12,
        pad=10,
    )
    ax.grid(axis="y", alpha=0.23)
    ax.grid(axis="x", alpha=0.12)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    cbar.set_label("label-chain position (%)")
    cbar.set_ticks([0, 25, 50, 75, 100])

    for i, run in enumerate(annotated_runs):
        x = (run.start_step + run.end_step) / 2.0
        y = run.chain_percent
        offset = 9 if (i % 2 == 0 or y < 18.0) and y < 95.0 else -12
        va = "bottom" if offset > 0 else "top"
        color = colors[min(max(run.start_step - steps[0], 0), len(colors) - 1)]
        label_text = wrap_label(run.label, width=20)
        ax.annotate(
            f"{label_text}\n{y:.1f}%",
            xy=(x, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=7.2,
            color="#0f172a",
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.8,
                "alpha": 0.9,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.7,
                "alpha": 0.75,
            },
            clip_on=True,
            zorder=3,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return True


def iter_metric_csvs(path):
    path = Path(path)
    if path.is_file():
        yield path
        return

    yield from sorted(path.rglob("metrics_over_time.csv"))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the label-chain position percentage of the top softmax label "
            "for metrics_over_time.csv files."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Single metrics_over_time.csv file to plot.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Directory to search recursively for metrics_over_time.csv files.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="top_softmax_label_chain_percent.png",
        help="Output PNG name saved next to each CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output PNGs.",
    )
    parser.add_argument(
        "--max-annotations",
        type=int,
        default=36,
        help="Maximum labeled top-label runs to annotate per plot.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print one progress line after this many CSV files. Use 1 for every file.",
    )
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.csv) == bool(args.root):
        raise SystemExit("Provide exactly one of --csv or --root.")

    source = args.csv if args.csv is not None else args.root
    csv_paths = list(iter_metric_csvs(source))
    if not csv_paths:
        raise SystemExit(f"No metrics_over_time.csv files found under {source}")

    written = 0
    skipped = 0
    failed = 0
    for i, csv_path in enumerate(csv_paths, start=1):
        output_path = csv_path.with_name(args.output_name)
        try:
            did_write = plot_top_label_chain_percent(
                csv_path=csv_path,
                output_path=output_path,
                overwrite=args.overwrite,
                max_annotations=args.max_annotations,
                dpi=args.dpi,
            )
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {csv_path}: {exc}")
            continue

        if did_write:
            written += 1
            action = "WROTE"
        else:
            skipped += 1
            action = "SKIP"
        should_print = (
            len(csv_paths) == 1
            or args.progress_every <= 1
            or i == 1
            or i == len(csv_paths)
            or i % args.progress_every == 0
        )
        if should_print:
            print(
                f"[{action}] {output_path} "
                f"({i}/{len(csv_paths)}, written={written}, skipped={skipped})"
            )

    print(f"Done. written={written} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
