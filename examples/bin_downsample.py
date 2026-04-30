"""Example driver: per-tile bin downsampling with fq-tile-subsample.

Generic Python wrapper that runs fq-tile-subsample across a set of paired
FASTQ samples at multiple bin sizes, then (optionally) plots summary
metrics from the generated manifests.

Usage:
    # Run downsampling on a set of samples
    python bin_downsample.py downsample \\
        --binary target/release/fq-tile-subsample \\
        --outdir /path/to/output \\
        --samples samples.tsv \\
        --bin-sizes 30000 20000 10000 5000 2500 1000 500

    # Plot results (reads the per-sample manifest.json files)
    python bin_downsample.py plot --outdir /path/to/output

samples.tsv format (tab-separated, one sample per line):
    sample_id    r1_path    r2_path

Each sample produces:
    <outdir>/<sample_id>/
        B<bin_size>/<r1_basename>          # downsampled FASTQs
        B<bin_size>/<r2_basename>
        manifest.json                      # per-bin-size stats
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_BIN_SIZES = [30000, 20000, 10000, 5000, 2500, 1000, 500]


def load_samples(path: Path) -> list[tuple[str, Path, Path | None]]:
    """Parse a TSV with columns: sample_id, r1_path, [r2_path]."""
    samples = []
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            sid = row[0]
            r1 = Path(row[1])
            r2 = Path(row[2]) if len(row) >= 3 and row[2] else None
            samples.append((sid, r1, r2))
    log.info("loaded %d samples from %s", len(samples), path)
    return samples


def run_sample(
    binary: Path,
    sample_id: str,
    r1: Path,
    r2: Path | None,
    outdir: Path,
    bin_sizes: list[int],
    seed: int,
    temp_dir: Path | None,
) -> Path:
    """Run fq-tile-subsample for one sample. Returns the manifest path."""
    sample_outdir = outdir / sample_id
    sample_outdir.mkdir(parents=True, exist_ok=True)
    manifest = sample_outdir / "manifest.json"

    if manifest.exists():
        log.info("[%s] manifest exists, skipping (%s)", sample_id, manifest)
        return manifest

    cmd = [
        str(binary),
        str(r1),
    ]
    if r2:
        cmd.append(str(r2))
    cmd += ["-n", *[str(b) for b in bin_sizes]]
    cmd += ["-o", str(sample_outdir)]
    cmd += ["--seed", str(seed)]
    cmd += ["--manifest", str(manifest)]
    if temp_dir:
        cmd += ["--temp-dir", str(temp_dir)]

    log.info("[%s] running: %s", sample_id, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(
            "[%s] failed (exit %d)\nstdout: %s\nstderr: %s",
            sample_id,
            result.returncode,
            result.stdout,
            result.stderr,
        )
        raise RuntimeError(f"fq-tile-subsample failed for {sample_id}")
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            log.info("[%s] %s", sample_id, line)
    return manifest


def cmd_downsample(args: argparse.Namespace) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.samples)

    for sid, r1, r2 in samples:
        run_sample(
            binary=args.binary,
            sample_id=sid,
            r1=r1,
            r2=r2,
            outdir=args.outdir,
            bin_sizes=args.bin_sizes,
            seed=args.seed,
            temp_dir=args.temp_dir,
        )
    log.info("done: %d samples processed", len(samples))


def cmd_plot(args: argparse.Namespace) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.error("matplotlib is required for plotting: pip install matplotlib")
        sys.exit(1)

    manifests = sorted(args.outdir.glob("*/manifest.json"))
    if not manifests:
        log.error("no manifest.json files found under %s", args.outdir)
        sys.exit(1)

    # Aggregate manifest entries across samples
    rows = []
    for manifest in manifests:
        sample_id = manifest.parent.name
        with open(manifest) as f:
            entries = json.load(f)
        for e in entries:
            rows.append({"sample_id": sample_id, **e})

    log.info("loaded %d manifest rows across %d samples", len(rows), len(manifests))

    # Group by sample
    by_sample: dict[str, list[dict]] = {}
    for r in rows:
        by_sample.setdefault(r["sample_id"], []).append(r)
    for entries in by_sample.values():
        entries.sort(key=lambda e: e["bin_size"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for sid, entries in sorted(by_sample.items()):
        bins = [e["bin_size"] for e in entries]
        written = [e["records_written"] for e in entries]
        retained = [e["tiles_retained"] for e in entries]
        axes[0].plot(bins, written, "o-", label=sid, markersize=4)
        axes[1].plot(bins, retained, "o-", label=sid, markersize=4)

    axes[0].set(
        xlabel="Bin size (reads/tile)",
        ylabel="Records written",
        title="Records written vs bin size",
        xscale="log",
        yscale="log",
    )
    axes[1].set(
        xlabel="Bin size (reads/tile)",
        ylabel="Tiles retained",
        title="Tiles retained vs bin size",
        xscale="log",
    )
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=2)

    out_path = args.outdir / "bin_downsample_summary.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    log.info("wrote %s", out_path)


def main() -> None:
    description = (__doc__ or "").split("\n\n")[0]
    parser = argparse.ArgumentParser(description=description)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ds = sub.add_parser("downsample", help="Run fq-tile-subsample for each sample")
    p_ds.add_argument(
        "--binary",
        type=Path,
        required=True,
        help="Path to the fq-tile-subsample binary",
    )
    p_ds.add_argument(
        "--outdir", type=Path, required=True, help="Output directory root"
    )
    p_ds.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="TSV with columns: sample_id, r1_path, [r2_path]",
    )
    p_ds.add_argument(
        "--bin-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_BIN_SIZES,
        help=f"Per-tile bin sizes (default: {DEFAULT_BIN_SIZES})",
    )
    p_ds.add_argument("--seed", type=int, default=42)
    p_ds.add_argument("--temp-dir", type=Path, default=None)
    p_ds.set_defaults(func=cmd_downsample)

    p_plot = sub.add_parser("plot", help="Plot summary metrics from manifests")
    p_plot.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory root (same as downsample --outdir)",
    )
    p_plot.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
