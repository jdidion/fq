use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use flate2::{Compression, write::GzEncoder};
use fq::fastq::{
    self, Record,
    io::{IndexedReader, Writer},
};
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
    rngs::SmallRng,
};
use serde::Serialize;
use tempfile::TempDir;
use tracing::{info, warn};

/// Split a FASTQ file by Illumina tile, then subsample at multiple bin sizes.
///
/// Reads are split into per-tile temporary files once, then each bin size
/// is sampled from the pre-split tiles without re-reading the source.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Read 1 source (gzipped or plain FASTQ).
    r1_src: PathBuf,

    /// Read 2 source (gzipped or plain FASTQ).
    r2_src: Option<PathBuf>,

    /// Per-tile bin sizes to sample (number of reads per tile).
    /// Tiles with fewer reads than a given size are excluded at that level.
    #[arg(short = 'n', long, required = true, num_args = 1..)]
    bin_sizes: Vec<u64>,

    /// Output directory. Each bin size produces <outdir>/B<size>/<original_filename>.
    #[arg(short, long)]
    outdir: PathBuf,

    /// Seed for the random number generator. Ensures reproducible output.
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Directory for per-tile temp files. Defaults to system temp.
    #[arg(long)]
    temp_dir: Option<PathBuf>,

    /// Write a JSON manifest of outputs to this path.
    #[arg(long)]
    manifest: Option<PathBuf>,
}

#[derive(Serialize)]
struct ManifestEntry {
    bin_size: u64,
    r1: String,
    r2: Option<String>,
    input_records: usize,
    input_tiles: usize,
    tiles_retained: usize,
    tiles_discarded: usize,
    records_written: usize,
    downsample_rate: f64,
}

/// Packed (lane, tile) key.
fn parse_tile_bin(name: &[u8]) -> Option<u64> {
    let name = if name.first() == Some(&b'@') {
        &name[1..]
    } else {
        name
    };
    let name = name.split(|&b| b == b' ').next()?;
    let mut parts = name.split(|&b| b == b':');
    parts.next()?; // instrument
    parts.next()?; // run
    parts.next()?; // flowcell
    let lane: u32 = std::str::from_utf8(parts.next()?).ok()?.parse().ok()?;
    let tile: u32 = std::str::from_utf8(parts.next()?).ok()?.parse().ok()?;
    Some((lane as u64) << 32 | (tile as u64))
}

struct TileFiles {
    record_count: usize,
    r1_path: PathBuf,
    r2_path: Option<PathBuf>,
}

/// First pass: split source FASTQs into per-tile temp files.
fn split_by_tile(
    r1_src: &Path,
    r2_src: Option<&Path>,
    temp_dir: &Path,
) -> Result<(Vec<TileFiles>, u64)> {
    let mut r1 =
        fastq::fs::open(r1_src).with_context(|| format!("opening {}", r1_src.display()))?;
    let mut r1_rec = Record::default();

    let mut r2_reader = r2_src
        .map(|p| fastq::fs::open(p).with_context(|| format!("opening {}", p.display())))
        .transpose()?;
    let mut r2_rec = Record::default();

    let paired = r2_src.is_some();

    struct TileWriter {
        r1: Writer<BufWriter<File>>,
        r1_path: PathBuf,
        r2: Option<Writer<BufWriter<File>>>,
        r2_path: Option<PathBuf>,
        count: usize,
    }

    let mut writers: HashMap<u64, TileWriter> = HashMap::new();
    let mut parse_failures: u64 = 0;

    loop {
        let r1_len = r1.read_record(&mut r1_rec)?;
        let r2_len = if let Some(r2) = r2_reader.as_mut() {
            r2.read_record(&mut r2_rec)?
        } else {
            0
        };

        match (r1_len, r2_len, paired) {
            (0, 0, _) | (0, _, false) => break,
            (0, _, true) => bail!("R1 ended before R2"),
            (_, 0, true) => bail!("R2 ended before R1"),
            _ => {}
        }

        if let Some(bin_key) = parse_tile_bin(r1_rec.name()) {
            if !writers.contains_key(&bin_key) {
                let r1_path = temp_dir.join(format!("{bin_key}.r1.fq"));
                let r1_file = BufWriter::new(
                    File::create(&r1_path)
                        .with_context(|| format!("creating {}", r1_path.display()))?,
                );
                let (r2_w, r2_p) = if paired {
                    let p = temp_dir.join(format!("{bin_key}.r2.fq"));
                    let f = BufWriter::new(
                        File::create(&p).with_context(|| format!("creating {}", p.display()))?,
                    );
                    (Some(Writer::new(f)), Some(p))
                } else {
                    (None, None)
                };
                writers.insert(
                    bin_key,
                    TileWriter {
                        r1: Writer::new(r1_file),
                        r1_path,
                        r2: r2_w,
                        r2_path: r2_p,
                        count: 0,
                    },
                );
            }
            let tw = writers.get_mut(&bin_key).unwrap();

            tw.r1.write_record(&r1_rec)?;
            if let Some(w) = tw.r2.as_mut() {
                w.write_record(&r2_rec)?;
            }
            tw.count += 1;
        } else {
            parse_failures += 1;
        }
    }

    let tiles: Vec<TileFiles> = writers
        .into_values()
        .map(|tw| TileFiles {
            record_count: tw.count,
            r1_path: tw.r1_path,
            r2_path: tw.r2_path,
        })
        .collect();

    Ok((tiles, parse_failures))
}

fn is_gzipped(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("gz")
}

/// Create an output writer that gzips if the path ends in .gz.
fn create_output(path: &Path) -> io::Result<Box<dyn Write>> {
    let file = BufWriter::new(File::create(path)?);
    if is_gzipped(path) {
        Ok(Box::new(GzEncoder::new(file, Compression::default())))
    } else {
        Ok(Box::new(file))
    }
}

/// Sample nested index sets from a single tile. `targets` is a slice of bin
/// sizes the tile is retained at (bin_sizes with tile.record_count >= bin).
/// The largest set is drawn once, then each smaller set is derived as a
/// uniform subset of the next larger one, so outputs are nested.
///
/// Returns per-target selected indices in the caller's order.
fn sample_tile_nested(
    tile: &TileFiles,
    targets: &[usize],
    rng: &mut SmallRng,
) -> Result<Vec<Vec<usize>>> {
    let actual = tile.record_count;
    if actual == 0 || targets.is_empty() {
        return Ok(targets.iter().map(|_| Vec::new()).collect());
    }

    let largest_target = targets.iter().copied().max().unwrap_or(0).min(actual);
    if largest_target == 0 {
        return Ok(targets.iter().map(|_| Vec::new()).collect());
    }

    let dist = Uniform::new(0, actual).unwrap();
    let mut selected = vec![false; actual];
    let mut n = 0;
    while n < largest_target {
        let i = dist.sample(rng);
        if !selected[i] {
            selected[i] = true;
            n += 1;
        }
    }
    let largest_indices: Vec<usize> = selected
        .iter()
        .enumerate()
        .filter(|&(_, s)| *s)
        .map(|(i, _)| i)
        .collect();

    // Sort targets descending to derive each smaller set from the previous.
    let mut order: Vec<usize> = (0..targets.len()).collect();
    order.sort_by_key(|i| std::cmp::Reverse(targets[*i]));

    let mut per_target: Vec<Vec<usize>> = vec![Vec::new(); targets.len()];
    let mut current = largest_indices;
    for idx in order {
        let want = targets[idx].min(current.len());
        while current.len() > want {
            let d = Uniform::new(0, current.len()).unwrap();
            let j = d.sample(rng);
            current.swap_remove(j);
        }
        let mut sorted = current.clone();
        sorted.sort_unstable();
        per_target[idx] = sorted;
    }

    Ok(per_target)
}

/// Read records at the given indices for the tile and write them to the
/// provided output writers.
fn emit_tile_records(
    tile: &TileFiles,
    indices: &[usize],
    r1_out: &mut dyn Write,
    r2_out: &mut Option<Box<dyn Write>>,
) -> Result<()> {
    if indices.is_empty() {
        return Ok(());
    }
    let mut r1_reader = IndexedReader::open(&tile.r1_path)
        .with_context(|| format!("indexing {}", tile.r1_path.display()))?;
    r1_reader.read_records_at(indices, |record| {
        r1_out.write_all(record.as_ref())?;
        Ok(())
    })?;

    if let (Some(r2_path), Some(r2_w)) = (&tile.r2_path, r2_out.as_mut()) {
        let mut r2_reader = IndexedReader::open(r2_path)
            .with_context(|| format!("indexing {}", r2_path.display()))?;
        let r2_count = r2_reader.index().record_count();
        let r2_indices: Vec<usize> = indices.iter().copied().filter(|&i| i < r2_count).collect();
        r2_reader.read_records_at(&r2_indices, |record| {
            r2_w.write_all(record.as_ref())?;
            Ok(())
        })?;
    }

    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let r1_basename = args
        .r1_src
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let r2_basename = args
        .r2_src
        .as_ref()
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string());

    let temp_dir = if let Some(ref parent) = args.temp_dir {
        TempDir::new_in(parent)?
    } else {
        TempDir::new()?
    };
    info!(temp_dir = %temp_dir.path().display(), "created temp directory");

    info!("splitting by tile");
    let (tiles, parse_failures) =
        split_by_tile(&args.r1_src, args.r2_src.as_deref(), temp_dir.path())?;

    let total_records: usize = tiles.iter().map(|t| t.record_count).sum();
    let num_tiles = tiles.len();
    info!(total_records, parse_failures, num_tiles, "split complete");

    if parse_failures > 0 {
        warn!("{parse_failures} records had unparseable Illumina headers");
    }

    let mut bin_sizes = args.bin_sizes.clone();
    bin_sizes.sort_unstable();
    bin_sizes.dedup();
    // Keep ascending: sample_tile_nested derives smaller sets from larger ones,
    // but returns results in the order of the caller's `targets` slice.
    // (manifest entries are emitted in the order used here.)

    // Prepare one output writer pair per bin size. Kept open for the entire
    // pass so each tile can fan out into all retained outputs.
    let mut output_paths: Vec<(PathBuf, Option<PathBuf>)> = Vec::with_capacity(bin_sizes.len());
    let mut r1_writers: Vec<Box<dyn Write>> = Vec::with_capacity(bin_sizes.len());
    let mut r2_writers: Vec<Option<Box<dyn Write>>> = Vec::with_capacity(bin_sizes.len());
    for &bin_size in &bin_sizes {
        let out_dir = args.outdir.join(format!("B{bin_size}"));
        std::fs::create_dir_all(&out_dir)?;
        let r1_dst = out_dir.join(&r1_basename);
        let r2_dst = r2_basename.as_ref().map(|name| out_dir.join(name));
        let r1_w = create_output(&r1_dst)?;
        let r2_w: Option<Box<dyn Write>> = r2_dst
            .as_ref()
            .map(|dst| create_output(dst))
            .transpose()?;
        output_paths.push((r1_dst, r2_dst));
        r1_writers.push(r1_w);
        r2_writers.push(r2_w);
    }

    // One deterministic RNG derived from the caller's seed. We use the same
    // RNG across all bin sizes because nested sampling shares index draws.
    let mut rng = SmallRng::seed_from_u64(args.seed);

    let mut records_written_per_bin = vec![0usize; bin_sizes.len()];
    let mut tiles_retained_per_bin = vec![0usize; bin_sizes.len()];

    for tile in &tiles {
        let retained: Vec<(usize, usize)> = bin_sizes
            .iter()
            .enumerate()
            .filter(|&(_, &bs)| tile.record_count >= bs as usize)
            .map(|(i, &bs)| (i, bs as usize))
            .collect();
        if retained.is_empty() {
            continue;
        }
        for (i, _) in &retained {
            tiles_retained_per_bin[*i] += 1;
        }
        let targets: Vec<usize> = retained.iter().map(|(_, t)| *t).collect();
        let per_target_indices = sample_tile_nested(tile, &targets, &mut rng)?;

        for ((bin_idx, _), indices) in retained.iter().zip(per_target_indices.iter()) {
            records_written_per_bin[*bin_idx] += indices.len();
            emit_tile_records(
                tile,
                indices,
                &mut *r1_writers[*bin_idx],
                &mut r2_writers[*bin_idx],
            )?;
        }
    }

    for w in r1_writers.iter_mut() {
        w.flush()?;
    }
    for w in r2_writers.iter_mut().flatten() {
        w.flush()?;
    }

    let mut manifest: Vec<ManifestEntry> = Vec::with_capacity(bin_sizes.len());
    for (i, &bin_size) in bin_sizes.iter().enumerate() {
        let tiles_retained = tiles_retained_per_bin[i];
        let tiles_discarded = num_tiles - tiles_retained;
        let records_written = records_written_per_bin[i];
        let pct = if total_records > 0 {
            records_written as f64 / total_records as f64 * 100.0
        } else {
            0.0
        };
        info!(
            bin_size,
            tiles_retained,
            tiles_discarded,
            records_written,
            "{:.1}% of input",
            pct
        );
        let (r1_dst, r2_dst) = &output_paths[i];
        manifest.push(ManifestEntry {
            bin_size,
            r1: r1_dst.to_string_lossy().to_string(),
            r2: r2_dst.as_ref().map(|p| p.to_string_lossy().to_string()),
            input_records: total_records,
            input_tiles: num_tiles,
            tiles_retained,
            tiles_discarded,
            records_written,
            downsample_rate: if total_records > 0 {
                records_written as f64 / total_records as f64
            } else {
                0.0
            },
        });
    }

    if let Some(manifest_path) = &args.manifest {
        let f = File::create(manifest_path)?;
        serde_json::to_writer_pretty(f, &manifest)?;
        info!(manifest = %manifest_path.display(), "wrote manifest");
    } else {
        serde_json::to_writer_pretty(io::stdout(), &manifest)?;
        println!();
    }

    info!("done");
    Ok(())
}
