#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from make_validation_figure import analytical_ema_bytes, load_orojenesis


R = Path(__file__).resolve().parents[1]
FPDF = R / "figures" / "flops_sweep_ema_error.pdf"
FPNG = R / "figures" / "flops_sweep_ema_error.png"

FLOP_CONFIGS = [
    ("0.1 GFLOPs", 0.1e9),
    ("1 GFLOP", 1.0e9),
    ("10 GFLOPs", 10.0e9),
    ("100 GFLOPs", 100.0e9),
]
BASE_FLOPS = 2.0 * 4096**3


def scaled_curves(target_flops: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sram, oro_base = load_orojenesis()
    this_work_base = analytical_ema_bytes(sram)
    scale = target_flops / BASE_FLOPS
    oro = oro_base * scale
    this_work = this_work_base * scale
    error = np.abs(this_work - oro) / oro * 100.0
    return sram, oro, this_work, error


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), dpi=220, sharex=True)
    oro_color = "#1f5a7a"
    this_color = "#8f3d55"
    err_color = "#d07a22"

    legend_handles = None
    for ax, (title, flops) in zip(axes.flat, FLOP_CONFIGS):
        sram, oro, this_work, error = scaled_curves(flops)
        x_mib = sram / 2**20

        ax.set_xscale("log")
        ax.set_yscale("log")
        oro_line, = ax.plot(x_mib, oro / 2**20, color=oro_color, linewidth=2.0, label="Orojenesis")
        this_line, = ax.plot(x_mib, this_work / 2**20, color=this_color, linewidth=1.8, linestyle="--", label="This Work")
        ax.set_title(title)
        ax.set_ylabel("EMA (MiB)")
        ax.grid(True, which="both", alpha=0.22)

        twin = ax.twinx()
        err_line, = twin.plot(x_mib, error, color=err_color, linewidth=1.6, label="Relative error")
        twin.set_ylabel("Relative error (%)")
        twin.set_ylim(0.0, max(5.0, float(np.ceil(error.max() / 5.0) * 5.0)))
        twin.tick_params(axis="y", colors=err_color)
        twin.yaxis.label.set_color(err_color)

        if legend_handles is None:
            legend_handles = [oro_line, this_line, err_line]

    for ax in axes[-1, :]:
        ax.set_xlabel("SRAM capacity (MiB)")

    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.7)

    FPDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FPDF, bbox_inches="tight")
    fig.savefig(FPNG, bbox_inches="tight")
    print(f"Wrote {FPDF}")
    print(f"Wrote {FPNG}")


if __name__ == "__main__":
    main()
