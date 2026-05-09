#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


R = Path(__file__).resolve().parents[1]
TP = R / "third_party"
ZIP = TP / "orojenesis.zip"
AROOT = TP / "orojenesis"
AIN = AROOT / "orojenesis"
L = R / "logs"
F = R / "figures"
D = R / "data"
TJSON = L / "orojenesis_mm_same_flops_timing.json"
TJSONL = L / "orojenesis_runs.jsonl"
CSV_OUT = D / "orojenesis_mm_same_flops.csv"
FIG_OUT = F / "orojenesis_MM_same_flops_data_gpu.pdf"

GPU_DATA = {
    "simt": {
        "sram_bytes": [2 * 2**20, 24 * 2**20, 40 * 2**20, 50 * 2**20],
        "dram_access_bytes": [2.69 * 2**30, 650 * 2**20, 533 * 2**20, 373.39 * 2**20],
    },
    "tensor": {
        "sram_bytes": [2 * 2**20, 24 * 2**20, 40 * 2**20, 50 * 2**20],
        "dram_access_bytes": [1.58 * 2**30, 561.51 * 2**20, 411.06 * 2**20, 373.33 * 2**20],
    },
}


def extract_art() -> None:
    if AIN.exists():
        return
    if not ZIP.exists():
        raise FileNotFoundError(f"missing {ZIP}")
    AROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as zf:
        zf.extractall(AROOT)


def check_env() -> dict:
    base = os.environ.get("TIMELOOP_BASE_PATH")
    mapper = Path(base) / "bin" / "timeloop-mapper" if base else None
    return {
        "zip_exists": ZIP.exists(),
        "artifact_extracted": AIN.exists(),
        "TIMELOOP_BASE_PATH": base,
        "timeloop_mapper": str(mapper) if mapper else None,
        "timeloop_mapper_exists": mapper.exists() if mapper else False,
    }


def load_mods():
    os.chdir(AIN)
    sys.path.insert(0, str(AIN))
    import src.plots as plots  # type: ignore
    import src.utils as utils  # type: ignore

    return utils, plots


def save_csv(df, coef: float) -> None:
    D.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = out.index * coef
    out["DRAM_Accesses"] = out["DRAM_Accesses"] * coef
    out = out.reset_index().rename(columns={"index": "sram_bytes", "DRAM_Accesses": "dram_access_bytes"})
    out[["sram_bytes", "dram_access_bytes", "Op_Intensity", "mapping"]].to_csv(CSV_OUT, index=False)


def run_job(force_rerun: bool, check_only: bool) -> int:
    env = check_env()
    if check_only:
        print(json.dumps(env, indent=2, sort_keys=True))
        return 0 if env["timeloop_mapper_exists"] else 2
    extract_art()
    if not env["timeloop_mapper_exists"]:
        print(json.dumps(env, indent=2, sort_keys=True), file=sys.stderr)
        print("timeloop-mapper not found", file=sys.stderr)
        return 2

    L.mkdir(parents=True, exist_ok=True)
    F.mkdir(parents=True, exist_ok=True)
    (AIN / "figs").mkdir(parents=True, exist_ok=True)
    os.environ["TIMELOOP_ENABLE_FIRST_READ_ELISION"] = "1"

    utils, plots = load_mods()

    out = Path("./outputs/single-einsum")
    arch = Path("./configs/single-einsum/arch.yaml")
    map_yaml = Path("./configs/single-einsum/conv_mapper.yaml")
    p = utils.Conv(P=4096, C=4096, K=4096)
    pdir = out / p.to_str()
    pre = pdir / "timeloop-mapper.oaves.csv"
    post = pdir / "oaves.csv"
    before = {"preprocessed_csv_exists": pre.exists(), "postprocessed_csv_exists": post.exists()}

    t0 = time.perf_counter()
    ts0 = datetime.now(timezone.utc).isoformat()
    utils.GenerateBound(
        p,
        out,
        arch,
        map_yaml,
        keep_one_best_entry_across_buf=True,
        force_rerun=force_rerun,
    )
    dur = time.perf_counter() - t0
    ts1 = datetime.now(timezone.utc).isoformat()

    sts = utils.get_stats_files(out, [p])
    dfs = utils.get_dfs(sts, get_opt=True)
    coef = 4
    save_csv(dfs[0], coef)

    gpu = {
        "simt": [GPU_DATA["simt"]["sram_bytes"], GPU_DATA["simt"]["dram_access_bytes"]],
        "tensor": [GPU_DATA["tensor"]["sram_bytes"], GPU_DATA["tensor"]["dram_access_bytes"]],
    }
    plots.plot_dfs(
        dfs,
        legends=["4k_4k_4k"],
        dpi=300,
        logy=False,
        logx=True,
        shape_name="MM_same_flops",
        figsize=(2.5, 2.5),
        xlim=(10**6, 2 * 10**8),
        ylim=(0, 4 * 10**9),
        y_end_value=2 * 10**8,
        plot_gpu_data=True,
        gpu_data=gpu,
        coefficient=coef,
        legend_fontsize=8,
    )

    af = AIN / "figs" / "oaves_opMM_same_flops_data_gpu.pdf"
    if af.exists():
        shutil.copy2(af, FIG_OUT)

    rec = {
        "name": "orojenesis_mm_same_flops",
        "wall_start_utc": ts0,
        "wall_end_utc": ts1,
        "duration_seconds": dur,
        "force_rerun": force_rerun,
        "environment": env,
        "cache_before": before,
        "cache_after": {"preprocessed_csv_exists": pre.exists(), "postprocessed_csv_exists": post.exists()},
        "problem": {"P": 4096, "C": 4096, "K": 4096, "precision_bits": 8},
        "coefficient_from_official_notebook": coef,
        "official_gpu_data": GPU_DATA,
        "artifact_outputs": {
            "preprocessed_csv": str(pre.resolve()),
            "postprocessed_csv": str(post.resolve()),
            "artifact_figure": str(af.resolve()),
        },
        "exported_outputs": {
            "curve_csv": str(CSV_OUT.resolve()),
            "figure_pdf": str(FIG_OUT.resolve()),
            "timing_json": str(TJSON.resolve()),
        },
    }
    TJSON.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    with TJSONL.open("a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()
    code = run_job(force_rerun=args.force_rerun, check_only=args.check_env)
    if code:
        sys.exit(code)
    return 0


if __name__ == "__main__":
    main()
