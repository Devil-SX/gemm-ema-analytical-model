#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


R = Path(__file__).resolve().parents[1]
FPDF = R / "figures" / "ema_component_breakdown.pdf"
FPNG = R / "figures" / "ema_component_breakdown.png"

P_BITS = 32
SIZES = np.geomspace(256, 16384, 240)
SRAM_CONFIGS = [
    ("100 KiB", 100 * 1024 * 8),
    ("1 MiB", 1 * 1024 * 1024 * 8),
    ("10 MiB", 10 * 1024 * 1024 * 8),
    ("100 MiB", 100 * 1024 * 1024 * 8),
]


def component_ratios(sram_bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flops = 2.0 * SIZES**3
    stationary = SIZES * SIZES * P_BITS
    reload_term = 2.0 * np.sqrt((stationary * stationary * stationary) / sram_bits)
    total = stationary + reload_term
    return flops, stationary / total * 100.0, reload_term / total * 100.0


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), dpi=220, sharex=True, sharey=False)
    colors = ["#b9c8d8", "#285f7d"]
    labels = [r"stationary $\min(S_i,S_w,S_o)$", r"reload $2\sqrt{S_iS_wS_o/M}$"]

    for ax, (title, sram_bits) in zip(axes.flat, SRAM_CONFIGS):
        flops, stationary_ratio, reload_ratio = component_ratios(sram_bits)
        ax.stackplot(flops / 1e9, stationary_ratio, reload_ratio, colors=colors, labels=labels, alpha=0.95)
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.grid(True, which="both", axis="both", alpha=0.22)
        ax.tick_params(axis="y", labelleft=True)

    for ax in axes[:, 0]:
        ax.set_ylabel("EMA composition (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel("FLOPs (GFLOPs, log scale)")

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))

    FPDF.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FPDF, bbox_inches="tight")
    fig.savefig(FPNG, bbox_inches="tight")
    print(f"Wrote {FPDF}")
    print(f"Wrote {FPNG}")


if __name__ == "__main__":
    main()
