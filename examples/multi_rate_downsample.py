"""Example driver: multi-rate downsampling with fq subsample.

Demonstrates the `--probability`, `--record-count`, and
`--record-count-per-tile` flags being passed multiple values to produce
nested output subsets in a single pass. Smaller outputs are guaranteed
subsets of larger ones at the same rate class.

Usage:
    # Nested count subsets — one FASTQ pair per -n value.
    python multi_rate_downsample.py \\
        --binary target/release/fq \\
        --mode count \\
        --values 1000 10000 100000 1000000 \\
        --r1-src sample_R1.fq.gz --r2-src sample_R2.fq.gz \\
        --outdir out/

    # Nested probability subsets.
    python multi_rate_downsample.py \\
        --binary target/release/fq --mode probability \\
        --values 0.01 0.05 0.25 0.50 \\
        --r1-src sample_R1.fq.gz --outdir out/

    # Nested per-tile-bin subsets (Illumina headers required).
    python multi_rate_downsample.py \\
        --binary target/release/fq --mode per_tile \\
        --values 500 1000 5000 \\
        --r1-src sample_R1.fq.gz --r2-src sample_R2.fq.gz \\
        --outdir out/

Outputs land at:
    <outdir>/<sample>_{quantity}_R1.fq.gz
    <outdir>/<sample>_{quantity}_R2.fq.gz    # if paired
where `{quantity}` is `p05`, `n10K`, `t500`, etc. — the same short
labels fq writes by default.
"""

import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODE_TO_FLAG = {
    "probability": "--probability",
    "count": "--record-count",
    "per_tile": "--record-count-per-tile",
}


def main() -> None:
    description = (__doc__ or "").split("\n\n")[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--binary", type=Path, required=True, help="Path to the fq binary")
    parser.add_argument(
        "--mode",
        choices=MODE_TO_FLAG,
        required=True,
        help="Rate class: probability | count | per_tile",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        required=True,
        help="Rate values (floats for probability, ints otherwise)",
    )
    parser.add_argument("--r1-src", type=Path, required=True)
    parser.add_argument("--r2-src", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extension",
        default=".fq.gz",
        help="Output extension (`.fq` for plain, `.fq.gz` to gzip). Default: .fq.gz",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    sample = args.r1_src.name.split(".", 1)[0]

    r1_tpl = str(args.outdir / f"{sample}_{{quantity}}_R1{args.extension}")
    r2_tpl = (
        str(args.outdir / f"{sample}_{{quantity}}_R2{args.extension}")
        if args.r2_src
        else None
    )

    cmd: list[str] = [
        str(args.binary),
        "subsample",
        MODE_TO_FLAG[args.mode],
        *args.values,
        "--r1-dst-template",
        r1_tpl,
    ]
    if r2_tpl:
        cmd += ["--r2-dst-template", r2_tpl]
    cmd += ["--seed", str(args.seed), str(args.r1_src)]
    if args.r2_src:
        cmd.append(str(args.r2_src))

    log.info("running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    log.info("done — outputs under %s", args.outdir)


if __name__ == "__main__":
    main()
