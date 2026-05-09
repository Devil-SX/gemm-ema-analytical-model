#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_orojenesis_mm_same_flops import GPU_DATA


R = Path(__file__).resolve().parents[1]
CSV = R / "data" / "orojenesis_mm_same_flops.csv"
FPDF = R / "figures" / "validation_orojenesis_this_work.pdf"
FPNG = R / "figures" / "validation_orojenesis_this_work.png"
ERR = R / "data" / "validation_orojenesis_this_work_error.csv"

M = 4096
K = 4096
N = 4096
P = 32
X_LIMITS = (1.0e6, 2.0e8)
Y_LIMIT_GIB = 4.0e9 / 2**30


def analytical_ema_bytes(sram_bytes: np.ndarray) -> np.ndarray:
    s_i = float(M * K * P)
    s_w = float(K * N * P)
    s_o = float(M * N * P)
    m_bits = sram_bytes * 8.0
    return (min(s_i, s_w, s_o) + 2.0 * np.sqrt((s_i * s_w * s_o) / m_bits)) / 8.0


def load_orojenesis() -> tuple[np.ndarray, np.ndarray]:
    sram = []
    dram = []
    with CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            sram.append(float(row["sram_bytes"]))
            dram.append(float(row["dram_access_bytes"]))
    return np.asarray(sram), np.asarray(dram)


def write_error_csv(sram: np.ndarray, oro: np.ndarray, this_work: np.ndarray) -> float:
    err = np.abs(this_work - oro) / oro * 100.0
    mask = sram > 1024.0
    with ERR.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sram_bytes", "orojenesis_bytes", "this_work_bytes", "relative_error_percent"])
        for values in zip(sram, oro, this_work, err):
            writer.writerow([f"{values[0]:.0f}", f"{values[1]:.8f}", f"{values[2]:.8f}", f"{values[3]:.6f}"])
    return float(err[mask].max())


def main() -> None:
    sram, oro = load_orojenesis()
    tw = analytical_ema_bytes(sram)
    err = np.abs(tw - oro) / oro * 100.0
    write_error_csv(sram, oro, tw)
    view = (sram >= X_LIMITS[0]) & (sram <= X_LIMITS[1])
    mx = float(err[view].max()) if np.any(view) else float(err.max())

    c = {
        "oro": "#1f5a7a",
        "simt": "#d07a22",
        "tensor": "#2a8c60",
        "this": "#8f3d55",
        "err": "#4d4d4d",
    }

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.1, 2.7), dpi=220)

    oro_line, = ax0.plot(sram, oro / 2**30, color=c["oro"], linewidth=2.0, label="Orojenesis")
    simt_pts = ax0.scatter(
        GPU_DATA["simt"]["sram_bytes"],
        np.asarray(GPU_DATA["simt"]["dram_access_bytes"]) / 2**30,
        color=c["simt"],
        marker="o",
        s=24,
        label="SIMT",
        zorder=3,
    )
    for label, x_value, y_value in zip(
        ["A2", "A30", "A100", "H100"],
        GPU_DATA["simt"]["sram_bytes"],
        np.asarray(GPU_DATA["simt"]["dram_access_bytes"]) / 2**30,
    ):
        ax0.annotate(label, (x_value, y_value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7, color=c["simt"])
    ten_pts = ax0.scatter(
        GPU_DATA["tensor"]["sram_bytes"],
        np.asarray(GPU_DATA["tensor"]["dram_access_bytes"]) / 2**30,
        color=c["tensor"],
        marker="x",
        s=30,
        label="Tensor",
        zorder=3,
    )
    this_line, = ax0.plot(sram, tw / 2**30, color=c["this"], linestyle="--", linewidth=2.0, label="This Work")
    ax0.set_xscale("log", base=2)
    ax0.set_xlim(*X_LIMITS)
    ax0.set_ylim(0.0, Y_LIMIT_GIB)
    ax0.set_xlabel("SRAM capacity (B)")
    ax0.set_ylabel("External access (GiB)")
    ax0.set_title("SRAM-capacity vs. EMA")
    ax0.grid(True, which="both", alpha=0.25)
    ax0.legend(handles=[this_line, oro_line, simt_pts, ten_pts], frameon=False)

    ax1.plot(sram, err, color=c["err"], linewidth=2.0)
    ax1.set_xscale("log", base=2)
    ax1.set_xlim(*X_LIMITS)
    ax1.set_xlabel("SRAM capacity (B)")
    ax1.set_ylabel("Relative error (%)")
    ax1.set_title("This Work vs. Orojenesis")
    ax1.set_ylim(0.0, max(12.0, math.ceil(mx) + 1.0))
    ax1.grid(True, which="both", alpha=0.25)
    ax1.text(
        0.05,
        0.88,
        f"max error in view: {mx:.2f}%",
        transform=ax1.transAxes,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc"},
    )

    fig.tight_layout(w_pad=1.6)
    FPDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FPDF, bbox_inches="tight")
    fig.savefig(FPNG, bbox_inches="tight")
    print(f"Wrote {FPDF}")
    print(f"Wrote {FPNG}")
    print(f"Wrote {ERR}")
    print(f"Max relative error in plotted view: {mx:.2f}%")


if __name__ == "__main__":
    main()
