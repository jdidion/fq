use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    ops::{Bound, RangeBounds},
    path::{Path, PathBuf},
    sync::mpsc,
};

use bitvec::vec::BitVec;
use flate2::{Compression, bufread::MultiGzDecoder, write::GzEncoder};
use rand::{
    RngExt, SeedableRng,
    distr::{Distribution, Uniform},
    rngs::SmallRng,
};
use rand_distr::Poisson;
use tempfile::TempDir;
use thiserror::Error;
use tracing::{info, info_span, warn};

use crate::{
    cli::SubsampleArgs,
    fastq::{
        self, Record,
        io::{IndexedReader, RecordIndex},
    },
};

const VALID_PROBABILITY_RANGE: (Bound<f64>, Bound<f64>) =
    (Bound::Excluded(0.0), Bound::Excluded(1.0));

pub fn subsample(args: SubsampleArgs) -> Result<(), SubsampleError> {
    let r1_src = &args.r1_src;
    let r2_src = args.r2_src.as_ref();

    info!(command = "subsample", "fq");

    let want_tile = args.bin_by_tile || !args.record_count_per_tile.is_empty();

    // --with-replacement is only implemented for the streaming and exact
    // whole-file paths; reject the combinations we do not support up front.
    if args.with_replacement {
        if args.fast {
            return Err(SubsampleError::ReplacementUnsupported("--fast"));
        }
        if want_tile {
            return Err(SubsampleError::ReplacementUnsupported("tile binning"));
        }
    }

    if args.num_samples < 1 {
        return Err(SubsampleError::InvalidNumSamples);
    }
    // Multiple replicates need a template so each replicate gets a distinct path.
    if args.num_samples > 1 && args.r1_dst_template.is_none() {
        return Err(SubsampleError::NumSamplesNeedsTemplate);
    }

    let quantity = resolve_quantity(&args)?;

    for sample_idx in 0..args.num_samples {
        // Derive a per-replicate RNG so each replicate is independent yet, when
        // --seed is given, the whole run is reproducible.
        let rng = match args.seed {
            Some(seed) => {
                let derived =
                    seed.wrapping_add(u64::from(sample_idx).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                if sample_idx == 0 {
                    info!(seed, "initializing rng from seed");
                }
                SmallRng::seed_from_u64(derived)
            }
            None => {
                if sample_idx == 0 {
                    info!("initializing rng from entropy");
                }
                rand::make_rng()
            }
        };

        // Per-replicate destinations: when a template is in play, expand the
        // optional {sample} token so replicates do not clobber each other.
        let r1_tpl = args
            .r1_dst_template
            .as_ref()
            .map(|t| expand_sample(t, sample_idx, args.num_samples));
        let r2_tpl = args
            .r2_dst_template
            .as_ref()
            .map(|t| expand_sample(t, sample_idx, args.num_samples));

        let r1_dsts =
            resolve_destinations(args.r1_dst.as_deref(), r1_tpl.as_deref(), &quantity, "r1")?;
        let r2_dsts = if r2_src.is_some() {
            let dsts =
                resolve_destinations(args.r2_dst.as_deref(), r2_tpl.as_deref(), &quantity, "r2")?;
            Some(dsts)
        } else {
            if args.r2_dst.is_some() || args.r2_dst_template.is_some() {
                return Err(SubsampleError::MissingSource("r2-src"));
            }
            None
        };

        let r1_dsts: Vec<&Path> = r1_dsts.iter().map(|p| p.as_path()).collect();
        let r2_dsts_refs: Option<Vec<&Path>> = r2_dsts
            .as_ref()
            .map(|v| v.iter().map(|p| p.as_path()).collect());
        let r2 = (r2_src.map(|p| &**p), r2_dsts_refs.as_deref());

        if args.num_samples > 1 {
            info!(sample = sample_idx + 1, of = args.num_samples, "replicate");
        }

        if want_tile {
            let tile_mode = match &quantity {
                Quantity::Fraction(ps) => TileCountMode::FromFraction(ps.clone()),
                Quantity::RecordCount(ns) => TileCountMode::FromRecordCount(ns.clone()),
                Quantity::RecordCountPerTile(ns) => TileCountMode::Explicit(ns.clone()),
            };
            subsample_by_tile(
                (r1_src, &r1_dsts),
                r2,
                rng,
                tile_mode,
                args.fast,
                args.sampling_threads,
                args.compression_threads,
                args.in_memory,
                args.temp_dir.as_deref(),
            )?;
        } else {
            match &quantity {
                Quantity::Fraction(ps) => {
                    subsample_approximate((r1_src, &r1_dsts), r2, rng, ps, args.with_replacement)?;
                }
                Quantity::RecordCount(ns) => {
                    let can_skip_ahead =
                        args.fast && !is_gzipped(r1_src) && r2_src.is_none() && ns.len() == 1;
                    if args.fast && ns.len() > 1 {
                        return Err(SubsampleError::FastMultiRate);
                    }
                    if can_skip_ahead {
                        subsample_skip_ahead((r1_src, r1_dsts[0]), rng, ns[0])?;
                    } else {
                        subsample_exact((r1_src, &r1_dsts), r2, rng, ns, args.with_replacement)?;
                    }
                }
                Quantity::RecordCountPerTile(_) => unreachable!("handled by tile branch"),
            }
        }
    }

    info!("done");

    Ok(())
}

/// Expand the optional `{sample}` token in a destination template. When the
/// template has no `{sample}` token but multiple replicates are requested, a
/// `.sN` suffix is inserted before the recognized extension so replicates do
/// not collide. For a single replicate the template is returned unchanged.
fn expand_sample(template: &str, sample_idx: u32, num_samples: u32) -> String {
    if num_samples <= 1 {
        return template.to_string();
    }
    let label = format!("s{}", sample_idx + 1);
    if template.contains("{sample}") {
        return template.replace("{sample}", &label);
    }
    // No explicit token: insert `.sN` before the first extension separator so
    // `out/r1_{quantity}.fq.gz` becomes `out/r1_{quantity}.s1.fq.gz`.
    match template.find('.') {
        Some(dot) => format!("{}.{}{}", &template[..dot], label, &template[dot..]),
        None => format!("{template}.{label}"),
    }
}

/// The resolved quantity in canonical ascending order, paired with the
/// original request indices so we can map outputs back to the user's inputs.
#[derive(Debug, Clone)]
enum Quantity {
    Fraction(Vec<f64>),
    RecordCount(Vec<u64>),
    RecordCountPerTile(Vec<u64>),
}

impl Quantity {
    fn len(&self) -> usize {
        match self {
            Quantity::Fraction(v) => v.len(),
            Quantity::RecordCount(v) | Quantity::RecordCountPerTile(v) => v.len(),
        }
    }

    fn labels(&self) -> Vec<String> {
        match self {
            Quantity::Fraction(v) => v.iter().map(|p| format_probability_label(*p)).collect(),
            Quantity::RecordCount(v) => v.iter().map(|n| format_count_label('n', *n)).collect(),
            Quantity::RecordCountPerTile(v) => {
                v.iter().map(|n| format_count_label('t', *n)).collect()
            }
        }
    }

    fn values(&self) -> Vec<String> {
        match self {
            Quantity::Fraction(v) => v.iter().map(|p| format!("{p}")).collect(),
            Quantity::RecordCount(v) | Quantity::RecordCountPerTile(v) => {
                v.iter().map(|n| format!("{n}")).collect()
            }
        }
    }
}

fn resolve_quantity(args: &SubsampleArgs) -> Result<Quantity, SubsampleError> {
    let p_given = !args.fraction.is_empty();
    let n_given = !args.record_count.is_empty();
    let t_given = !args.record_count_per_tile.is_empty();

    match (p_given, n_given, t_given) {
        (true, false, false) => {
            let mut v = args.fraction.clone();
            for &f in &v {
                // Reject non-finite and non-positive fractions: a Poisson rate
                // must be finite and > 0.0 (this is what makes the later
                // Poisson::new in the streaming path infallible).
                if !f.is_finite() || f <= 0.0 {
                    return Err(SubsampleError::InvalidFraction(f));
                }
                // A fraction > 1.0 oversamples and only makes sense with
                // replacement; without replacement the range is (0.0, 1.0).
                if !args.with_replacement && !VALID_PROBABILITY_RANGE.contains(&f) {
                    return Err(SubsampleError::FractionRequiresReplacement(f));
                }
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.dedup();
            Ok(Quantity::Fraction(v))
        }
        (false, true, false) => {
            let mut v = args.record_count.clone();
            v.sort_unstable();
            v.dedup();
            Ok(Quantity::RecordCount(v))
        }
        (false, false, true) => {
            let mut v = args.record_count_per_tile.clone();
            v.sort_unstable();
            v.dedup();
            Ok(Quantity::RecordCountPerTile(v))
        }
        _ => unreachable!("CLI ArgGroup enforces exactly one quantity kind"),
    }
}

/// Maps a quantity to output paths, either by cloning a single `--dst` path
/// or by expanding a `--dst-template` once per value.
fn resolve_destinations(
    single: Option<&Path>,
    template: Option<&str>,
    quantity: &Quantity,
    label: &'static str,
) -> Result<Vec<PathBuf>, SubsampleError> {
    let labels = quantity.labels();
    let values = quantity.values();

    match (single, template) {
        (Some(path), None) => {
            if quantity.len() != 1 {
                return Err(SubsampleError::MultiRateNeedsTemplate(label));
            }
            Ok(vec![path.to_path_buf()])
        }
        (None, Some(tpl)) => {
            // A {quantity}/{value} token is only needed to disambiguate multiple
            // rates. A single-rate template (e.g. one driven solely by {sample}
            // for replicates) does not require it.
            if quantity.len() > 1 && !tpl.contains("{quantity}") && !tpl.contains("{value}") {
                return Err(SubsampleError::TemplateMissingToken(label));
            }
            let mut out = Vec::with_capacity(quantity.len());
            for (lab, val) in labels.iter().zip(values.iter()) {
                out.push(PathBuf::from(expand_template(tpl, lab, val)));
            }
            Ok(out)
        }
        (Some(_), Some(_)) => unreachable!("CLI enforces dst/template mutual exclusion"),
        (None, None) => Err(SubsampleError::MissingDestination(label)),
    }
}

fn expand_template(template: &str, label: &str, value: &str) -> String {
    template
        .replace("{quantity}", label)
        .replace("{value}", value)
}

/// Format a probability like 0.05 as "p05" and 0.5 as "p50".
fn format_probability_label(p: f64) -> String {
    let pct = (p * 100.0).round() as i64;
    if pct >= 0 && (p * 100.0 - pct as f64).abs() < 1e-9 {
        format!("p{pct:02}")
    } else {
        // fall back to decimal representation with dots swapped for underscores
        let s = format!("{p}").replace('.', "_");
        format!("p{s}")
    }
}

/// Format a record count as a compact suffix: 500 → "n500", 10_000 → "n10K",
/// 2_500_000 → "n2_5M". The `prefix` chooses the namespace ('n' or 't').
fn format_count_label(prefix: char, n: u64) -> String {
    if n == 0 {
        return format!("{prefix}0");
    }
    const UNITS: &[(u64, char)] = &[(1_000_000_000, 'G'), (1_000_000, 'M'), (1_000, 'K')];
    for &(div, sym) in UNITS {
        if n % div == 0 {
            return format!("{prefix}{}{sym}", n / div);
        }
    }
    format!("{prefix}{n}")
}

fn is_gzipped(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("gz")
}

fn subsample_approximate<Rng>(
    (r1_src, r1_dsts): (&Path, &[&Path]),
    (r2_src, r2_dsts): (Option<&Path>, Option<&[&Path]>),
    mut rng: Rng,
    fractions: &[f64],
    with_replacement: bool,
) -> Result<(), SubsampleError>
where
    Rng: rand::Rng,
{
    for &f in fractions {
        if !f.is_finite() || f <= 0.0 {
            return Err(SubsampleError::InvalidFraction(f));
        }
        if !with_replacement && !VALID_PROBABILITY_RANGE.contains(&f) {
            return Err(SubsampleError::FractionRequiresReplacement(f));
        }
    }
    assert_eq!(fractions.len(), r1_dsts.len());

    let mut r1 = fastq::fs::open(r1_src).map_err(|e| SubsampleError::OpenFile(e, r1_src.into()))?;
    let mut w1s: Vec<_> = r1_dsts
        .iter()
        .map(|dst| fastq::fs::create(dst).map_err(|e| SubsampleError::CreateFile(e, (*dst).into())))
        .collect::<Result<Vec<_>, _>>()?;

    let span = info_span!("subsample_approximate", rates = fractions.len());
    let _span_ctx = span.enter();

    let (ns, total) = match (r2_src, r2_dsts) {
        (Some(r2_src), Some(r2_dsts)) => {
            assert_eq!(r2_dsts.len(), fractions.len());
            info!("sampling paired end reads");

            let mut r2 =
                fastq::fs::open(r2_src).map_err(|e| SubsampleError::OpenFile(e, r2_src.into()))?;
            let mut w2s: Vec<_> = r2_dsts
                .iter()
                .map(|dst| {
                    fastq::fs::create(dst).map_err(|e| SubsampleError::CreateFile(e, (*dst).into()))
                })
                .collect::<Result<Vec<_>, _>>()?;

            subsample_paired_multi(
                (&mut r1, &mut w1s),
                (&mut r2, &mut w2s),
                &mut rng,
                fractions,
                with_replacement,
            )?
        }
        (Some(_), None) => return Err(SubsampleError::MissingDestination("r2-dst")),
        (None, Some(_)) => return Err(SubsampleError::MissingSource("r2-src")),
        _ => {
            info!("sampling single end reads");
            subsample_single_multi(&mut r1, &mut w1s, &mut rng, fractions, with_replacement)?
        }
    };

    for (f, n) in fractions.iter().zip(ns.iter()) {
        let percentage = (*n as f64) / (total as f64) * 100.0;
        info!(
            f,
            n, total, "sampled {}/{} ({:.1}%) records", n, total, percentage
        );
    }

    Ok(())
}

/// Precompute one Poisson(f) distribution per rate for the with-replacement
/// streaming path. Returns `Ok(None)` when sampling without replacement.
///
/// Callers validate that every `f` is finite and > 0.0, so `Poisson::new`
/// should not fail; the error is propagated rather than unwrapped so an
/// unvalidated caller surfaces a clean error instead of panicking.
fn poisson_per_rate(
    fs: &[f64],
    with_replacement: bool,
) -> Result<Option<Vec<Poisson<f64>>>, SubsampleError> {
    if !with_replacement {
        return Ok(None);
    }
    let dists = fs
        .iter()
        .map(|&f| Poisson::new(f).map_err(|_| SubsampleError::InvalidFraction(f)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(dists))
}

fn subsample_single_multi<R, W, Rng>(
    reader: &mut fastq::io::Reader<R>,
    writers: &mut [fastq::io::Writer<W>],
    rng: &mut Rng,
    fs: &[f64],
    with_replacement: bool,
) -> Result<(Vec<u64>, u64), SubsampleError>
where
    R: BufRead,
    W: Write,
    Rng: rand::Rng,
{
    let poissons = poisson_per_rate(fs, with_replacement)?;
    let mut record = Record::default();
    let mut counts = vec![0u64; fs.len()];
    let mut total = 0u64;

    while reader.read_record(&mut record)? != 0 {
        // Without replacement all rates share one Bernoulli draw, so smaller
        // rates are nested subsets of larger ones; the draw is only needed in
        // that branch. With replacement each rate draws its own Poisson count
        // (independent, no nesting guarantee).
        let q: Option<f64> = poissons.is_none().then(|| rng.random());
        for i in 0..fs.len() {
            let mult = match &poissons {
                Some(ps) => ps[i].sample(rng) as u64,
                None if q.unwrap() <= fs[i] => 1,
                None => 0,
            };
            for _ in 0..mult {
                writers[i].write_record(&record)?;
            }
            counts[i] += mult;
        }
        total += 1;
    }

    Ok((counts, total))
}

fn subsample_paired_multi<R, S, W, X, Rng>(
    (r1, w1s): (&mut fastq::io::Reader<R>, &mut [fastq::io::Writer<W>]),
    (r2, w2s): (&mut fastq::io::Reader<S>, &mut [fastq::io::Writer<X>]),
    rng: &mut Rng,
    fs: &[f64],
    with_replacement: bool,
) -> Result<(Vec<u64>, u64), SubsampleError>
where
    R: BufRead,
    S: BufRead,
    W: Write,
    X: Write,
    Rng: rand::Rng,
{
    let poissons = poisson_per_rate(fs, with_replacement)?;
    let mut s1 = Record::default();
    let mut s2 = Record::default();
    let mut counts = vec![0u64; fs.len()];
    let mut total = 0u64;

    loop {
        match (r1.read_record(&mut s1)?, r2.read_record(&mut s2)?) {
            (0, 0) => break,
            (0, len) if len > 0 => return Err(SubsampleError::UnexpectedEof("r1-src")),
            (len, 0) if len > 0 => return Err(SubsampleError::UnexpectedEof("r2-src")),
            (_, _) => {
                // One draw per pair, governing both mates, keeps R1/R2 in sync.
                let q: Option<f64> = poissons.is_none().then(|| rng.random());
                for i in 0..fs.len() {
                    let mult = match &poissons {
                        Some(ps) => ps[i].sample(rng) as u64,
                        None if q.unwrap() <= fs[i] => 1,
                        None => 0,
                    };
                    for _ in 0..mult {
                        w1s[i].write_record(&s1)?;
                        w2s[i].write_record(&s2)?;
                    }
                    counts[i] += mult;
                }
                total += 1;
            }
        }
    }

    Ok((counts, total))
}

fn subsample_exact<Rng>(
    (r1_src, r1_dsts): (&Path, &[&Path]),
    (r2_src, r2_dsts): (Option<&Path>, Option<&[&Path]>),
    rng: Rng,
    record_counts: &[u64],
    with_replacement: bool,
) -> Result<(), SubsampleError>
where
    Rng: rand::Rng,
{
    assert_eq!(record_counts.len(), r1_dsts.len());
    let span = info_span!("subsample_exact", rates = record_counts.len());
    let _span_ctx = span.enter();

    info!("counting records");

    let actual_record_count = if let Some(fai_count) = RecordIndex::from_fai(r1_src)? {
        if is_gzipped(r1_src) {
            info!(
                actual_record_count = fai_count,
                "counted records from .fai index (gzipped; not cross-checked)"
            );
            fai_count
        } else {
            let index = RecordIndex::build_from_path(r1_src)?;
            let file_count = index.record_count();
            if fai_count != file_count {
                warn!(
                    ".fai record count ({}) differs from file record count ({}); using file count (index may be stale)",
                    fai_count, file_count
                );
            }
            info!(
                actual_record_count = file_count,
                "counted records (verified against .fai)"
            );
            file_count
        }
    } else if !is_gzipped(r1_src) {
        let index = RecordIndex::build_from_path(r1_src)?;
        let count = index.record_count();
        info!(actual_record_count = count, "counted records");
        count
    } else {
        let line_count = count_lines(r1_src)?;
        let count = line_count / 4;
        info!(actual_record_count = count, "counted records (line scan)");
        count
    };

    if actual_record_count == 0 {
        info!("input is empty; producing empty output");
        for r1_dst in r1_dsts {
            fastq::fs::create(r1_dst)
                .map_err(|e| SubsampleError::CreateFile(e, (*r1_dst).into()))?;
        }
        if let Some(r2_dsts) = r2_dsts {
            for r2_dst in r2_dsts {
                fastq::fs::create(r2_dst)
                    .map_err(|e| SubsampleError::CreateFile(e, (*r2_dst).into()))?;
            }
        }
        return Ok(());
    }

    let n_available = u64::try_from(actual_record_count)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    // Without replacement a request can never exceed the available records, so
    // clamp it. With replacement records may repeat, so any count is valid
    // (counts > n_available oversample).
    let effective_counts: Vec<u64> = record_counts
        .iter()
        .map(|&rc| {
            if !with_replacement && rc > n_available {
                warn!(
                    "record count ({}) > r1-src record count ({}). Using record-count = {} instead.",
                    rc, n_available, n_available
                );
                n_available
            } else {
                rc
            }
        })
        .collect();

    info!("building selection multiplicities");
    // Without replacement the selection is a set of 1-bit-per-record bitmaps
    // (cheap); with replacement it is a per-record count vector (records may
    // repeat). Keeping the two representations distinct avoids materializing the
    // bitmaps into wider count vectors just to share an emit routine.
    let selection = if with_replacement {
        Selection::Counts(build_with_replacement(
            rng,
            actual_record_count,
            &effective_counts,
        )?)
    } else {
        Selection::Bits(build_nested_filters(
            rng,
            actual_record_count,
            &effective_counts,
        )?)
    };
    info!("built multiplicities");

    let mut r1 = fastq::fs::open(r1_src).map_err(|e| SubsampleError::OpenFile(e, r1_src.into()))?;
    let mut w1s: Vec<_> = r1_dsts
        .iter()
        .map(|dst| fastq::fs::create(dst).map_err(|e| SubsampleError::CreateFile(e, (*dst).into())))
        .collect::<Result<Vec<_>, _>>()?;

    match (r2_src, r2_dsts) {
        (Some(r2_src), Some(r2_dsts)) => {
            assert_eq!(r2_dsts.len(), record_counts.len());
            info!("sampling paired end reads");

            let mut r2 =
                fastq::fs::open(r2_src).map_err(|e| SubsampleError::OpenFile(e, r2_src.into()))?;
            let mut w2s: Vec<_> = r2_dsts
                .iter()
                .map(|dst| {
                    fastq::fs::create(dst).map_err(|e| SubsampleError::CreateFile(e, (*dst).into()))
                })
                .collect::<Result<Vec<_>, _>>()?;

            subsample_exact_paired_multi((&mut r1, &mut w1s), (&mut r2, &mut w2s), &selection)?;
        }
        (Some(_), None) => return Err(SubsampleError::MissingDestination("r2-dst")),
        (None, Some(_)) => return Err(SubsampleError::MissingSource("r2-src")),
        (None, None) => {
            info!("sampling single end reads");
            subsample_exact_single_multi(&mut r1, &mut w1s, &selection)?;
        }
    }

    // The number of emitted records is the sum of the multiplicities, which
    // equals the requested count for both modes (with replacement it may exceed
    // the input size).
    for (rate, _) in r1_dsts.iter().enumerate() {
        let emitted = selection.emitted(rate);
        let percentage = (emitted as f64) / (actual_record_count as f64) * 100.0;
        info!(
            emitted,
            total = actual_record_count,
            "sampled {}/{} ({:.1}%) records",
            emitted,
            actual_record_count,
            percentage
        );
    }

    Ok(())
}

/// The per-rate record selection for the exact path, in one of two
/// representations:
///
/// * `Bits` — without replacement: one 1-bit-per-record bitmap per rate. A
///   record is emitted at most once, so multiplicity is 0 or 1.
/// * `Counts` — with replacement: one count-per-record vector per rate. A
///   record may be emitted any number of times, so counts are `u64` to allow
///   extreme oversampling without overflow.
enum Selection {
    Bits(Vec<BitVec>),
    Counts(Vec<Vec<u64>>),
}

impl Selection {
    /// The number of rates (output destinations).
    fn len(&self) -> usize {
        match self {
            Selection::Bits(v) => v.len(),
            Selection::Counts(v) => v.len(),
        }
    }

    /// How many times record `i` should be written for output `rate`.
    fn multiplicity(&self, rate: usize, i: usize) -> u64 {
        match self {
            Selection::Bits(v) => u64::from(v[rate][i]),
            Selection::Counts(v) => v[rate][i],
        }
    }

    /// Total records emitted for `rate` (the sum of its multiplicities).
    fn emitted(&self, rate: usize) -> u64 {
        match self {
            Selection::Bits(v) => v[rate].count_ones() as u64,
            Selection::Counts(v) => v[rate].iter().sum(),
        }
    }
}

/// Build per-rate multiplicity vectors by sampling `count` record indices
/// uniformly *with replacement* from `0..src_record_count`. A record's
/// multiplicity is the number of times its index was drawn, so records may
/// repeat and `count` may exceed `src_record_count` (oversampling). Each rate is
/// drawn independently; there is no nesting guarantee.
fn build_with_replacement<Rng>(
    mut rng: Rng,
    src_record_count: usize,
    counts: &[u64],
) -> Result<Vec<Vec<u64>>, SubsampleError>
where
    Rng: rand::Rng,
{
    let dist = Uniform::new(0, src_record_count).map_err(SubsampleError::InvalidUniformRange)?;
    let mut out = Vec::with_capacity(counts.len());
    for &count in counts {
        let mut mult = vec![0u64; src_record_count];
        for _ in 0..count {
            let i = dist.sample(&mut rng);
            mult[i] += 1;
        }
        out.push(mult);
    }
    Ok(out)
}

fn count_lines<P>(src: P) -> io::Result<usize>
where
    P: AsRef<Path>,
{
    const LINE_FEED: u8 = b'\n';

    let mut reader = open_maybe_gz(src)?;
    let mut n = 0;

    loop {
        let buf = reader.fill_buf()?;

        if buf.is_empty() {
            break;
        }

        n += bytecount::count(buf, LINE_FEED);

        let len = buf.len();
        reader.consume(len);
    }

    Ok(n)
}

fn open_maybe_gz<P>(src: P) -> io::Result<Box<dyn BufRead>>
where
    P: AsRef<Path>,
{
    let path = src.as_ref();
    let extension = path.extension();
    let reader = File::open(path).map(BufReader::new)?;

    match extension.and_then(|ext| ext.to_str()) {
        Some("gz") => {
            let decoder = MultiGzDecoder::new(reader);
            Ok(Box::new(BufReader::new(decoder)))
        }
        _ => Ok(Box::new(reader)),
    }
}

fn build_filter<Rng>(
    mut rng: Rng,
    src_record_count: usize,
    dst_record_count: u64,
) -> Result<BitVec, SubsampleError>
where
    Rng: rand::Rng,
{
    let mut bitmap = BitVec::new();
    bitmap.resize(src_record_count, false);

    let distribution =
        Uniform::new(0, src_record_count).map_err(SubsampleError::InvalidUniformRange)?;

    let mut n = 0;

    while n < dst_record_count {
        let i = distribution.sample(&mut rng);

        if !bitmap[i] {
            bitmap.set(i, true);
            n += 1;
        }
    }

    Ok(bitmap)
}

/// Build a nested sequence of bitmaps for the provided target counts (which
/// must already be sorted ascending). `bitmaps[i]` always contains
/// `record_counts[i]` set bits and is a subset of `bitmaps[j]` for all j > i.
///
/// Strategy: build the largest bitmap via rejection sampling, then derive
/// each smaller bitmap by uniformly sampling set-bits of the next larger one.
fn build_nested_filters<Rng>(
    mut rng: Rng,
    src_record_count: usize,
    record_counts: &[u64],
) -> Result<Vec<BitVec>, SubsampleError>
where
    Rng: rand::Rng,
{
    let n = record_counts.len();
    debug_assert!(record_counts.windows(2).all(|w| w[0] <= w[1]));

    if n == 0 {
        return Ok(Vec::new());
    }

    let mut bitmaps: Vec<BitVec> = Vec::with_capacity(n);
    let largest = *record_counts.last().unwrap();
    let largest_bm = build_filter(&mut rng, src_record_count, largest)?;
    bitmaps.push(largest_bm);

    // Build smaller bitmaps by downsampling the previous one.
    // We iterate from the largest down to the smallest.
    for i in (0..n - 1).rev() {
        let target = record_counts[i];
        let prev = bitmaps.last().unwrap();
        let prev_indices: Vec<usize> = prev.iter_ones().collect();
        let prev_count = prev_indices.len() as u64;
        let target = target.min(prev_count);

        let mut next = BitVec::new();
        next.resize(src_record_count, false);

        if prev_count == 0 || target == 0 {
            bitmaps.push(next);
            continue;
        }

        let dist =
            Uniform::new(0, prev_indices.len()).map_err(SubsampleError::InvalidUniformRange)?;
        let mut selected = 0u64;
        let mut seen = vec![false; prev_indices.len()];
        while selected < target {
            let j = dist.sample(&mut rng);
            if !seen[j] {
                seen[j] = true;
                next.set(prev_indices[j], true);
                selected += 1;
            }
        }

        bitmaps.push(next);
    }

    bitmaps.reverse();
    Ok(bitmaps)
}

#[cfg(test)]
fn subsample_exact_single<R, W>(
    reader: &mut fastq::io::Reader<R>,
    writer: &mut fastq::io::Writer<W>,
    bitmap: &BitVec,
) -> Result<(), SubsampleError>
where
    R: BufRead,
    W: Write,
{
    let mut record = Record::default();
    let mut i = 0;

    while reader.read_record(&mut record)? != 0 {
        if bitmap[i] {
            writer.write_record(&record)?;
        }

        i += 1;
    }

    Ok(())
}

fn subsample_exact_single_multi<R, W>(
    reader: &mut fastq::io::Reader<R>,
    writers: &mut [fastq::io::Writer<W>],
    selection: &Selection,
) -> Result<(), SubsampleError>
where
    R: BufRead,
    W: Write,
{
    assert_eq!(writers.len(), selection.len());
    let mut record = Record::default();
    let mut i = 0;

    while reader.read_record(&mut record)? != 0 {
        for (rate, w) in writers.iter_mut().enumerate() {
            for _ in 0..selection.multiplicity(rate, i) {
                w.write_record(&record)?;
            }
        }
        i += 1;
    }

    Ok(())
}

#[cfg(test)]
fn subsample_exact_paired<R, S, W, X>(
    (r1, w1): (&mut fastq::io::Reader<R>, &mut fastq::io::Writer<W>),
    (r2, w2): (&mut fastq::io::Reader<S>, &mut fastq::io::Writer<X>),
    bitmap: &BitVec,
) -> Result<(), SubsampleError>
where
    R: BufRead,
    S: BufRead,
    W: Write,
    X: Write,
{
    let mut s1 = Record::default();
    let mut s2 = Record::default();

    let mut i = 0;

    loop {
        match (r1.read_record(&mut s1)?, r2.read_record(&mut s2)?) {
            (0, 0) => break,
            (0, len) if len > 0 => return Err(SubsampleError::UnexpectedEof("r1-src")),
            (len, 0) if len > 0 => return Err(SubsampleError::UnexpectedEof("r2-src")),
            (_, _) => {
                if bitmap[i] {
                    w1.write_record(&s1)?;
                    w2.write_record(&s2)?;
                }

                i += 1;
            }
        }
    }

    Ok(())
}

fn subsample_exact_paired_multi<R, S, W, X>(
    (r1, w1s): (&mut fastq::io::Reader<R>, &mut [fastq::io::Writer<W>]),
    (r2, w2s): (&mut fastq::io::Reader<S>, &mut [fastq::io::Writer<X>]),
    selection: &Selection,
) -> Result<(), SubsampleError>
where
    R: BufRead,
    S: BufRead,
    W: Write,
    X: Write,
{
    assert_eq!(w1s.len(), selection.len());
    assert_eq!(w2s.len(), selection.len());

    let mut s1 = Record::default();
    let mut s2 = Record::default();
    let mut i = 0;

    loop {
        match (r1.read_record(&mut s1)?, r2.read_record(&mut s2)?) {
            (0, 0) => break,
            (0, len) if len > 0 => return Err(SubsampleError::UnexpectedEof("r1-src")),
            (len, 0) if len > 0 => return Err(SubsampleError::UnexpectedEof("r2-src")),
            (_, _) => {
                // The same multiplicity governs both mates, keeping R1/R2 in sync.
                for (rate, (w1, w2)) in w1s.iter_mut().zip(w2s.iter_mut()).enumerate() {
                    for _ in 0..selection.multiplicity(rate, i) {
                        w1.write_record(&s1)?;
                        w2.write_record(&s2)?;
                    }
                }
                i += 1;
            }
        }
    }

    Ok(())
}

fn subsample_skip_ahead<Rng>(
    (r1_src, r1_dst): (&Path, &Path),
    mut rng: Rng,
    target_count: u64,
) -> Result<(), SubsampleError>
where
    Rng: rand::Rng,
{
    let span = info_span!("subsample_skip_ahead", target_count);
    let _span_ctx = span.enter();

    let mut r1_reader =
        IndexedReader::open(r1_src).map_err(|e| SubsampleError::OpenFile(e, r1_src.into()))?;

    let total = r1_reader.index().record_count();
    info!(total_records = total, "built index");

    let target = (target_count as usize).min(total);
    let mut r1_buf = Vec::new();

    let selected = r1_reader.skip_ahead_sample(target, &mut rng, |record| {
        r1_buf.extend_from_slice(record.as_ref());
        Ok(())
    })?;

    info!(selected, "skip-ahead sampling complete");

    write_output_data(&r1_buf, r1_dst, 1)?;

    let percentage = if total > 0 {
        selected as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    info!(
        "sampled ~{}/{} ({:.1}%) records",
        selected, total, percentage
    );

    Ok(())
}

enum TileCountMode {
    Explicit(Vec<u64>),
    FromRecordCount(Vec<u64>),
    FromFraction(Vec<f64>),
}

/// Parses an Illumina read header to extract (lane, tile) as a packed u64 key.
///
/// Expected format: `@<instrument>:<run>:<flowcell>:<lane>:<tile>:<x>:<y>`
fn parse_tile_bin(name: &[u8]) -> Option<u64> {
    let name = if name.first() == Some(&b'@') {
        &name[1..]
    } else {
        name
    };

    // Strip description (everything after first space)
    let name = name.split(|&b| b == b' ').next()?;

    let mut parts = name.split(|&b| b == b':');

    // Skip instrument, run, flowcell (fields 0-2)
    parts.next()?;
    parts.next()?;
    parts.next()?;

    let lane_bytes = parts.next()?;
    let tile_bytes = parts.next()?;

    let lane: u32 = std::str::from_utf8(lane_bytes).ok()?.parse().ok()?;
    let tile: u32 = std::str::from_utf8(tile_bytes).ok()?.parse().ok()?;

    Some((lane as u64) << 32 | (tile as u64))
}

struct TileResult {
    r1_data: Vec<u8>,
    r2_data: Option<Vec<u8>>,
}

enum OutputWriter {
    Plain(BufWriter<File>),
    Gz(GzEncoder<BufWriter<File>>),
}

impl OutputWriter {
    fn create(dst: &Path) -> io::Result<Self> {
        let file = BufWriter::new(File::create(dst)?);
        if is_gzipped(dst) {
            Ok(OutputWriter::Gz(GzEncoder::new(
                file,
                Compression::default(),
            )))
        } else {
            Ok(OutputWriter::Plain(file))
        }
    }

    fn finish(self) -> io::Result<()> {
        match self {
            OutputWriter::Plain(mut w) => w.flush(),
            OutputWriter::Gz(w) => {
                w.finish()?;
                Ok(())
            }
        }
    }
}

impl Write for OutputWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            OutputWriter::Plain(w) => w.write(buf),
            OutputWriter::Gz(w) => w.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            OutputWriter::Plain(w) => w.flush(),
            OutputWriter::Gz(w) => w.flush(),
        }
    }
}

struct TileInfo {
    bin_key: u64,
    record_count: usize,
    r1_path: PathBuf,
    r2_path: Option<PathBuf>,
}

struct TileWriter {
    r1: fastq::io::Writer<BufWriter<File>>,
    r1_path: PathBuf,
    r2: Option<fastq::io::Writer<BufWriter<File>>>,
    r2_path: Option<PathBuf>,
    record_count: usize,
}

fn write_tile_temp_files(
    r1_src: &Path,
    r2_src: Option<&Path>,
    temp_dir: &Path,
) -> Result<(Vec<TileInfo>, u64), SubsampleError> {
    let mut r1 = fastq::fs::open(r1_src).map_err(|e| SubsampleError::OpenFile(e, r1_src.into()))?;
    let mut r1_rec = Record::default();

    let mut r2_reader = r2_src
        .map(|p| fastq::fs::open(p).map_err(|e| SubsampleError::OpenFile(e, p.into())))
        .transpose()?;
    let mut r2_rec = Record::default();

    let paired = r2_src.is_some();
    let mut tile_writers: HashMap<u64, TileWriter> = HashMap::new();
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
            (0, _, true) => return Err(SubsampleError::UnexpectedEof("r1-src")),
            (_, 0, true) => return Err(SubsampleError::UnexpectedEof("r2-src")),
            _ => {}
        }

        if let Some(bin_key) = parse_tile_bin(r1_rec.name()) {
            if !tile_writers.contains_key(&bin_key) {
                let r1_path = temp_dir.join(format!("{bin_key}.r1.fq"));
                let r1_file = BufWriter::new(
                    File::create(&r1_path)
                        .map_err(|e| SubsampleError::CreateFile(e, r1_path.clone()))?,
                );
                let (r2_writer, r2_path) = if paired {
                    let p = temp_dir.join(format!("{bin_key}.r2.fq"));
                    let f = BufWriter::new(
                        File::create(&p).map_err(|e| SubsampleError::CreateFile(e, p.clone()))?,
                    );
                    (Some(fastq::io::Writer::new(f)), Some(p))
                } else {
                    (None, None)
                };
                tile_writers.insert(
                    bin_key,
                    TileWriter {
                        r1: fastq::io::Writer::new(r1_file),
                        r1_path,
                        r2: r2_writer,
                        r2_path,
                        record_count: 0,
                    },
                );
            }
            let tw = tile_writers.get_mut(&bin_key).unwrap();

            tw.r1.write_record(&r1_rec)?;
            if let Some(r2w) = tw.r2.as_mut() {
                r2w.write_record(&r2_rec)?;
            }
            tw.record_count += 1;
        } else {
            parse_failures += 1;
        }
    }

    // Collect TileInfo (writers are dropped here, flushing buffers)
    let mut tiles: Vec<TileInfo> = tile_writers
        .into_iter()
        .map(|(key, tw)| TileInfo {
            bin_key: key,
            record_count: tw.record_count,
            r1_path: tw.r1_path,
            r2_path: tw.r2_path,
        })
        .collect();
    // Sort by bin key for deterministic processing order
    tiles.sort_unstable_by_key(|t| t.bin_key);

    Ok((tiles, parse_failures))
}

fn subsample_by_tile<Rng>(
    (r1_src, r1_dsts): (&Path, &[&Path]),
    (r2_src, r2_dsts): (Option<&Path>, Option<&[&Path]>),
    mut rng: Rng,
    mode: TileCountMode,
    fast: bool,
    sampling_threads: usize,
    compression_threads: usize,
    in_memory: bool,
    temp_dir_path: Option<&Path>,
) -> Result<(), SubsampleError>
where
    Rng: rand::Rng + Send,
{
    let span = info_span!("subsample_by_tile");
    let _span_ctx = span.enter();

    // Validate paired args
    match (r2_src, r2_dsts) {
        (Some(_), None) => return Err(SubsampleError::MissingDestination("r2-dst")),
        (None, Some(_)) => return Err(SubsampleError::MissingSource("r2-src")),
        _ => {}
    }

    if in_memory {
        return subsample_by_tile_in_memory(
            (r1_src, r1_dsts),
            (r2_src, r2_dsts),
            rng,
            mode,
            compression_threads,
        );
    }

    // Create temp directory
    let temp_dir = if let Some(parent) = temp_dir_path {
        TempDir::new_in(parent).map_err(SubsampleError::TempDir)?
    } else {
        TempDir::new().map_err(SubsampleError::TempDir)?
    };
    info!(temp_dir = %temp_dir.path().display(), "created temp directory");

    // First pass: write per-tile temp files
    info!("first pass: writing per-tile temp files");
    let (tiles, parse_failures) = write_tile_temp_files(r1_src, r2_src, temp_dir.path())?;

    let total_records: usize = tiles.iter().map(|t| t.record_count).sum();
    let num_bins = tiles.len();

    info!(
        total_records,
        parse_failures,
        bins = num_bins,
        "binning complete"
    );

    if parse_failures > 0 {
        warn!(
            "{} records had headers that could not be parsed as Illumina format",
            parse_failures
        );
    }

    // Compute per-tile targets, sorted ascending (nested).
    let mut per_tile_targets = compute_per_tile_targets(&mode, num_bins, total_records);
    per_tile_targets.sort_unstable();
    per_tile_targets.dedup();

    assert_eq!(
        per_tile_targets.len(),
        r1_dsts.len(),
        "expected one per-tile target per destination"
    );

    let max_target = *per_tile_targets.iter().max().unwrap_or(&0);
    info!(?per_tile_targets, "computed per-tile targets");

    if max_target == 0 {
        warn!("per-tile record count is 0; output will be empty");
        for dst in r1_dsts {
            write_output_data(&[], dst, compression_threads)?;
        }
        if let Some(r2_dsts) = r2_dsts {
            for dst in r2_dsts {
                write_output_data(&[], dst, compression_threads)?;
            }
        }
        return Ok(());
    }

    // Each rate keeps its own set of retained tiles (bins with >= target).
    let paired = r2_src.is_some();
    let threads = sampling_threads.min(num_bins.max(1)).max(1);

    info!(
        fast,
        sampling_threads = threads,
        rates = per_tile_targets.len(),
        "sampling tiles across {} rates",
        per_tile_targets.len()
    );

    let base_seed: u64 = rng.random();

    // Channel sends (rate_index, tile_result) so the writer thread can fan out.
    let (tx, rx) = mpsc::sync_channel::<(usize, TileResult)>(threads * 2);

    let r1_dsts_vec: Vec<PathBuf> = r1_dsts.iter().map(|p| p.to_path_buf()).collect();
    let r2_dsts_vec: Option<Vec<PathBuf>> =
        r2_dsts.map(|v| v.iter().map(|p| p.to_path_buf()).collect());

    let selected_totals = std::thread::scope(|s| -> Result<Vec<usize>, SubsampleError> {
        // Writer thread: receives (rate_index, data) and dispatches to the right writer pair.
        let r1_paths = r1_dsts_vec.clone();
        let r2_paths = r2_dsts_vec.clone();
        let writer_handle = s.spawn(move || -> Result<Vec<usize>, SubsampleError> {
            let mut r1_writers: Vec<OutputWriter> = r1_paths
                .iter()
                .map(|p| OutputWriter::create(p))
                .collect::<io::Result<Vec<_>>>()?;
            let mut r2_writers: Option<Vec<OutputWriter>> = if paired {
                Some(
                    r2_paths
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|p| OutputWriter::create(p))
                        .collect::<io::Result<Vec<_>>>()?,
                )
            } else {
                None
            };
            let mut counts = vec![0usize; r1_writers.len()];

            for (rate_idx, result) in rx {
                counts[rate_idx] += result.r1_data.iter().filter(|&&b| b == b'\n').count() / 4;
                r1_writers[rate_idx].write_all(&result.r1_data)?;
                if let (Some(ws), Some(data)) = (r2_writers.as_mut(), result.r2_data) {
                    ws[rate_idx].write_all(&data)?;
                }
            }

            for w in r1_writers {
                w.finish()?;
            }
            if let Some(ws) = r2_writers {
                for w in ws {
                    w.finish()?;
                }
            }

            Ok(counts)
        });

        // Build the per-rate tile work items. Each retained tile produces
        // (rate_idx, tile, target) — we sample the largest target once and
        // derive smaller ones by re-using the indexed reader.
        let chunk_size = (num_bins + threads - 1) / threads.max(1);
        let tile_chunks: Vec<&[TileInfo]> = tiles.chunks(chunk_size.max(1)).collect();
        let targets_arc = std::sync::Arc::new(per_tile_targets.clone());

        let handles: Vec<_> = tile_chunks
            .into_iter()
            .map(|chunk| {
                let tx = tx.clone();
                let targets = targets_arc.clone();
                s.spawn(move || -> Result<(), SubsampleError> {
                    for tile in chunk {
                        // Determine which rates retain this tile.
                        let retained_rates: Vec<usize> = targets
                            .iter()
                            .enumerate()
                            .filter(|(_, t)| tile.record_count >= **t as usize)
                            .map(|(i, _)| i)
                            .collect();
                        if retained_rates.is_empty() {
                            continue;
                        }

                        let mut tile_rng =
                            SmallRng::seed_from_u64(base_seed.wrapping_add(tile.bin_key));

                        // Sample the largest retained target, then derive smaller
                        // ones as subsets of the selected records.
                        let largest =
                            *retained_rates.iter().map(|i| &targets[*i]).max().unwrap() as usize;

                        if fast && !paired {
                            // Skip-ahead path: only supports a single target
                            // and doesn't provide per-record indices, so fall
                            // back to running it once per rate.
                            for &rate_idx in &retained_rates {
                                let target = targets[rate_idx] as usize;
                                let (r1_data, r2_data) =
                                    sample_tile_skip_ahead(tile, target, &mut tile_rng)?;
                                tx.send((rate_idx, TileResult { r1_data, r2_data }))
                                    .unwrap();
                            }
                        } else {
                            let nested = sample_tile_exact_nested(
                                tile,
                                &retained_rates
                                    .iter()
                                    .map(|i| targets[*i] as usize)
                                    .collect::<Vec<_>>(),
                                largest,
                                &mut tile_rng,
                            )?;
                            for (slot_idx, (r1_data, r2_data)) in nested.into_iter().enumerate() {
                                let rate_idx = retained_rates[slot_idx];
                                tx.send((rate_idx, TileResult { r1_data, r2_data }))
                                    .unwrap();
                            }
                        }
                    }
                    Ok(())
                })
            })
            .collect();

        drop(tx);

        for handle in handles {
            handle.join().unwrap()?;
        }

        writer_handle.join().unwrap()
    })?;

    for (target, total) in per_tile_targets.iter().zip(selected_totals.iter()) {
        let percentage = if total_records > 0 {
            *total as f64 / total_records as f64 * 100.0
        } else {
            0.0
        };
        info!(
            per_tile_target = target,
            total,
            total_records,
            "sampled {}/{} ({:.1}%) records",
            total,
            total_records,
            percentage
        );
    }

    Ok(())
}

/// Compute one per-tile target per requested rate, in the order the rates
/// appeared in `mode`. The caller sorts/dedups for nested ordering.
fn compute_per_tile_targets(
    mode: &TileCountMode,
    num_bins: usize,
    total_records: usize,
) -> Vec<u64> {
    match mode {
        TileCountMode::Explicit(v) => v.clone(),
        TileCountMode::FromRecordCount(v) => v
            .iter()
            .map(|target| {
                if num_bins == 0 {
                    0
                } else {
                    (*target as usize / num_bins) as u64
                }
            })
            .collect(),
        TileCountMode::FromFraction(v) => v
            .iter()
            .map(|p| {
                if num_bins == 0 {
                    0
                } else {
                    (p * total_records as f64 / num_bins as f64).floor() as u64
                }
            })
            .collect(),
    }
}

struct InMemoryTile {
    r1_records: Vec<Vec<u8>>,
    r2_records: Vec<Vec<u8>>,
}

fn subsample_by_tile_in_memory<Rng>(
    (r1_src, r1_dsts): (&Path, &[&Path]),
    (r2_src, r2_dsts): (Option<&Path>, Option<&[&Path]>),
    mut rng: Rng,
    mode: TileCountMode,
    compression_threads: usize,
) -> Result<(), SubsampleError>
where
    Rng: rand::Rng,
{
    info!("using in-memory tile binning");

    let mut r1 = fastq::fs::open(r1_src).map_err(|e| SubsampleError::OpenFile(e, r1_src.into()))?;
    let mut r1_rec = Record::default();

    let mut r2_reader = r2_src
        .map(|p| fastq::fs::open(p).map_err(|e| SubsampleError::OpenFile(e, p.into())))
        .transpose()?;
    let mut r2_rec = Record::default();

    let paired = r2_src.is_some();
    let mut tiles: HashMap<u64, InMemoryTile> = HashMap::new();
    let mut parse_failures: u64 = 0;

    // First pass: read all records into memory, binned by tile
    loop {
        let r1_len = r1.read_record(&mut r1_rec)?;
        let r2_len = if let Some(r2) = r2_reader.as_mut() {
            r2.read_record(&mut r2_rec)?
        } else {
            0
        };

        match (r1_len, r2_len, paired) {
            (0, 0, _) | (0, _, false) => break,
            (0, _, true) => return Err(SubsampleError::UnexpectedEof("r1-src")),
            (_, 0, true) => return Err(SubsampleError::UnexpectedEof("r2-src")),
            _ => {}
        }

        if let Some(bin_key) = parse_tile_bin(r1_rec.name()) {
            let tile = tiles.entry(bin_key).or_insert_with(|| InMemoryTile {
                r1_records: Vec::new(),
                r2_records: Vec::new(),
            });
            tile.r1_records.push(r1_rec.as_ref().to_vec());
            if paired {
                tile.r2_records.push(r2_rec.as_ref().to_vec());
            }
        } else {
            parse_failures += 1;
        }
    }

    let total_records: usize = tiles.values().map(|t| t.r1_records.len()).sum();
    let num_bins = tiles.len();

    info!(
        total_records,
        parse_failures,
        bins = num_bins,
        "in-memory binning complete"
    );

    if parse_failures > 0 {
        warn!(
            "{} records had headers that could not be parsed as Illumina format",
            parse_failures
        );
    }

    let mut per_tile_targets = compute_per_tile_targets(&mode, num_bins, total_records);
    per_tile_targets.sort_unstable();
    per_tile_targets.dedup();

    assert_eq!(per_tile_targets.len(), r1_dsts.len());

    info!(?per_tile_targets, "computed per-tile targets");

    let max_target = *per_tile_targets.iter().max().unwrap_or(&0);
    if max_target == 0 {
        warn!("per-tile record count is 0; output will be empty");
        for dst in r1_dsts {
            write_output_data(&[], dst, compression_threads)?;
        }
        if let Some(r2_dsts) = r2_dsts {
            for dst in r2_dsts {
                write_output_data(&[], dst, compression_threads)?;
            }
        }
        return Ok(());
    }

    // One output buffer per rate.
    let n_rates = per_tile_targets.len();
    let mut r1_bufs: Vec<Vec<u8>> = vec![Vec::new(); n_rates];
    let mut r2_bufs: Vec<Vec<u8>> = vec![Vec::new(); n_rates];

    for tile in tiles.values() {
        let count = tile.r1_records.len();
        // Find which rates retain this tile.
        let retained_rates: Vec<usize> = per_tile_targets
            .iter()
            .enumerate()
            .filter(|(_, t)| count >= **t as usize)
            .map(|(i, _)| i)
            .collect();
        if retained_rates.is_empty() {
            continue;
        }

        // Build nested index sets: largest set first, then derive smaller ones.
        let largest_target = *retained_rates
            .iter()
            .map(|i| &per_tile_targets[*i])
            .max()
            .unwrap() as usize;

        let distribution = Uniform::new(0, count).map_err(SubsampleError::InvalidUniformRange)?;
        let mut selected = vec![false; count];
        let mut n = 0;
        while n < largest_target {
            let i = distribution.sample(&mut rng);
            if !selected[i] {
                selected[i] = true;
                n += 1;
            }
        }
        let selected_indices: Vec<usize> = selected
            .iter()
            .enumerate()
            .filter(|&(_, s)| *s)
            .map(|(i, _)| i)
            .collect();

        // Sort retained_rates by target descending, then carry a running
        // index set down to each smaller target.
        let mut retained_sorted = retained_rates.clone();
        retained_sorted.sort_by_key(|i| std::cmp::Reverse(per_tile_targets[*i]));

        let mut current: Vec<usize> = selected_indices;
        for rate_idx in retained_sorted {
            let target = per_tile_targets[rate_idx] as usize;
            // Shrink `current` uniformly down to `target` items if needed.
            while current.len() > target {
                let dist =
                    Uniform::new(0, current.len()).map_err(SubsampleError::InvalidUniformRange)?;
                let j = dist.sample(&mut rng);
                current.swap_remove(j);
            }

            for &i in &current {
                r1_bufs[rate_idx].extend_from_slice(&tile.r1_records[i]);
                if paired {
                    r2_bufs[rate_idx].extend_from_slice(&tile.r2_records[i]);
                }
            }
        }
    }

    for (buf, dst) in r1_bufs.iter().zip(r1_dsts.iter()) {
        write_output_data(buf, dst, compression_threads)?;
    }
    if paired {
        let r2_dsts = r2_dsts.unwrap();
        for (buf, dst) in r2_bufs.iter().zip(r2_dsts.iter()) {
            write_output_data(buf, dst, compression_threads)?;
        }
    }

    for (target, buf) in per_tile_targets.iter().zip(r1_bufs.iter()) {
        let selected_records = buf.iter().filter(|&&b| b == b'\n').count() / 4;
        let percentage = if total_records > 0 {
            selected_records as f64 / total_records as f64 * 100.0
        } else {
            0.0
        };
        info!(
            per_tile_target = target,
            "sampled {}/{} ({:.1}%) records", selected_records, total_records, percentage
        );
    }

    Ok(())
}

fn sample_tile_skip_ahead<Rng>(
    tile: &TileInfo,
    target: usize,
    rng: &mut Rng,
) -> Result<(Vec<u8>, Option<Vec<u8>>), SubsampleError>
where
    Rng: rand::Rng,
{
    // Skip-ahead is single-end only (enforced by caller).
    // Uses IndexedReader to jump between known record offsets.
    let mut r1_reader = IndexedReader::open(&tile.r1_path)
        .map_err(|e| SubsampleError::OpenFile(e, tile.r1_path.clone()))?;

    if r1_reader.index().record_count() == 0 {
        return Ok((Vec::new(), None));
    }

    let mut r1_buf = Vec::new();
    r1_reader.skip_ahead_sample(target, rng, |record| {
        r1_buf.extend_from_slice(record.as_ref());
        Ok(())
    })?;

    Ok((r1_buf, None))
}

/// Sample nested index sets from a single tile. `targets` lists the desired
/// count for each rate (in the caller's rate order). `largest_target` is the
/// maximum of `targets` — the largest set is drawn first, then each smaller
/// set is derived as a uniform subset of the next larger set. Returns the
/// per-rate (r1_data, r2_data) in the same order as `targets`.
fn sample_tile_exact_nested<Rng>(
    tile: &TileInfo,
    targets: &[usize],
    largest_target: usize,
    rng: &mut Rng,
) -> Result<Vec<(Vec<u8>, Option<Vec<u8>>)>, SubsampleError>
where
    Rng: rand::Rng,
{
    let mut r1_reader = IndexedReader::open(&tile.r1_path)
        .map_err(|e| SubsampleError::OpenFile(e, tile.r1_path.clone()))?;
    let actual_count = r1_reader.index().record_count();
    let largest_target = largest_target.min(actual_count);

    if actual_count == 0 || largest_target == 0 {
        return Ok(targets
            .iter()
            .map(|_| (Vec::new(), tile.r2_path.as_ref().map(|_| Vec::new())))
            .collect());
    }

    // Draw the largest set.
    let distribution =
        Uniform::new(0, actual_count).map_err(SubsampleError::InvalidUniformRange)?;
    let mut selected: Vec<bool> = vec![false; actual_count];
    let mut n = 0;
    while n < largest_target {
        let i = distribution.sample(rng);
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

    // Order rates largest-first and carry a running index set down.
    let mut order: Vec<usize> = (0..targets.len()).collect();
    order.sort_by_key(|i| std::cmp::Reverse(targets[*i]));

    let mut per_rate_indices: Vec<Vec<usize>> = vec![Vec::new(); targets.len()];
    let mut current = largest_indices;
    for rate_idx in &order {
        let want = targets[*rate_idx].min(current.len());
        while current.len() > want {
            let dist =
                Uniform::new(0, current.len()).map_err(SubsampleError::InvalidUniformRange)?;
            let j = dist.sample(rng);
            current.swap_remove(j);
        }
        let mut sorted = current.clone();
        sorted.sort_unstable();
        per_rate_indices[*rate_idx] = sorted;
    }

    // Read all R1 records in one pass using the union (largest set) to avoid
    // opening the reader repeatedly, then slice per-rate into buffers.
    // For simplicity, we read per-rate with its own reader call — the file
    // is already local and small per tile.
    let mut r2_reader = if let Some(r2_path) = &tile.r2_path {
        Some(
            IndexedReader::open(r2_path)
                .map_err(|e| SubsampleError::OpenFile(e, r2_path.clone()))?,
        )
    } else {
        None
    };
    let r2_count = r2_reader
        .as_ref()
        .map(|r| r.index().record_count())
        .unwrap_or(0);

    let mut out: Vec<(Vec<u8>, Option<Vec<u8>>)> = Vec::with_capacity(targets.len());
    for rate_idx in 0..targets.len() {
        let indices = &per_rate_indices[rate_idx];
        let mut r1_buf = Vec::new();
        r1_reader.read_records_at(indices, |record| {
            r1_buf.extend_from_slice(record.as_ref());
            Ok(())
        })?;

        let r2_buf = if let Some(r2) = r2_reader.as_mut() {
            let r2_indices: Vec<usize> =
                indices.iter().copied().filter(|&i| i < r2_count).collect();
            let mut buf = Vec::new();
            r2.read_records_at(&r2_indices, |record| {
                buf.extend_from_slice(record.as_ref());
                Ok(())
            })?;
            Some(buf)
        } else {
            None
        };
        out.push((r1_buf, r2_buf));
    }

    Ok(out)
}

fn write_output_data(data: &[u8], dst: &Path, compression_threads: usize) -> io::Result<()> {
    if is_gzipped(dst) && compression_threads > 1 && !data.is_empty() {
        write_output_parallel_gz(data, dst, compression_threads)
    } else if is_gzipped(dst) {
        let file = BufWriter::new(File::create(dst)?);
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(data)?;
        encoder.finish()?;
        Ok(())
    } else {
        let mut file = BufWriter::new(File::create(dst)?);
        file.write_all(data)?;
        file.flush()?;
        Ok(())
    }
}

fn write_output_parallel_gz(data: &[u8], dst: &Path, threads: usize) -> io::Result<()> {
    let chunk_size = (data.len() + threads - 1) / threads;
    let chunks: Vec<&[u8]> = data.chunks(chunk_size).collect();

    let compressed_chunks: Vec<Vec<u8>> = std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                s.spawn(|| -> io::Result<Vec<u8>> {
                    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                    encoder.write_all(chunk)?;
                    encoder.finish()
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Result<Vec<_>, _>>()
    })?;

    let mut file = BufWriter::new(File::create(dst)?);
    for chunk in compressed_chunks {
        file.write_all(&chunk)?;
    }
    file.flush()?;

    Ok(())
}

#[derive(Debug, Error)]
pub enum SubsampleError {
    #[error("I/O error")]
    Io(#[from] io::Error),
    #[error("could not open file: {1}")]
    OpenFile(#[source] io::Error, PathBuf),
    #[error("could not create file: {1}")]
    CreateFile(#[source] io::Error, PathBuf),
    #[error("could not create temp directory")]
    TempDir(#[source] io::Error),
    #[error("missing pair source: {0}")]
    MissingSource(&'static str),
    #[error("missing pair destination: {0}")]
    MissingDestination(&'static str),
    #[error("invalid probability: expected (0.0, 1.0), got {0}")]
    InvalidProbability(f64),
    #[error("invalid fraction: expected a finite value > 0.0, got {0}")]
    InvalidFraction(f64),
    #[error("fraction {0} is outside (0.0, 1.0); a fraction >= 1.0 requires --with-replacement")]
    FractionRequiresReplacement(f64),
    #[error("--with-replacement is not supported with {0}")]
    ReplacementUnsupported(&'static str),
    #[error("--num-samples must be >= 1")]
    InvalidNumSamples,
    #[error("--num-samples > 1 requires --r1-dst-template")]
    NumSamplesNeedsTemplate,
    #[error("{0} unexpectedly ended")]
    UnexpectedEof(&'static str),
    #[error("invalid uniform range")]
    InvalidUniformRange(rand::distr::uniform::Error),
    #[error("--fast does not support multiple record counts")]
    FastMultiRate,
    #[error("multiple rates given for {0} but no --{0}-dst-template")]
    MultiRateNeedsTemplate(&'static str),
    #[error("--{0}-dst-template must contain {{quantity}} or {{value}}")]
    TemplateMissingToken(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subsample_single() -> Result<(), SubsampleError> {
        let data = b"@r1\nACGT\n+\nFQLB
@r2\nACGT\n+\nFQLB
@r3\nACGT\n+\nFQLB
@r4\nACGT\n+\nFQLB
";

        let mut reader = fastq::io::Reader::new(&data[..]);
        let mut writers = vec![fastq::io::Writer::new(Vec::new())];

        let mut rng = SmallRng::seed_from_u64(0);

        subsample_single_multi(&mut reader, &mut writers, &mut rng, &[0.33], false)?;

        let expected = b"@r1\nACGT\n+\nFQLB\n@r4\nACGT\n+\nFQLB\n";
        assert_eq!(writers[0].get_ref(), expected);

        Ok(())
    }

    #[test]
    fn test_subsample_paired() -> Result<(), SubsampleError> {
        let r1_data = b"@r1\nACGT\n+\nFQLB
@r2\nACGT\n+\nFQLB
@r3\nACGT\n+\nFQLB
@r4\nACGT\n+\nFQLB
";

        let r2_data = b"@r1\nTGCA\n+\nBLQF
@r2\nTGCA\n+\nBLQF
@r3\nTGCA\n+\nBLQF
@r4\nTGCA\n+\nBLQF
";

        let mut r1 = fastq::io::Reader::new(&r1_data[..]);
        let mut w1s = vec![fastq::io::Writer::new(Vec::new())];
        let mut r2 = fastq::io::Reader::new(&r2_data[..]);
        let mut w2s = vec![fastq::io::Writer::new(Vec::new())];

        let mut rng = SmallRng::seed_from_u64(0);

        subsample_paired_multi(
            (&mut r1, &mut w1s),
            (&mut r2, &mut w2s),
            &mut rng,
            &[0.33],
            false,
        )?;

        let w1_expected = b"@r1\nACGT\n+\nFQLB\n@r4\nACGT\n+\nFQLB\n";
        assert_eq!(w1s[0].get_ref(), w1_expected);

        let w2_expected = b"@r1\nTGCA\n+\nBLQF\n@r4\nTGCA\n+\nBLQF\n";
        assert_eq!(w2s[0].get_ref(), w2_expected);

        Ok(())
    }

    #[test]
    fn test_subsample_exact_single() -> Result<(), SubsampleError> {
        let data = b"@r1\nACGT\n+\nFQLB
@r2\nACGT\n+\nFQLB
@r3\nACGT\n+\nFQLB
@r4\nACGT\n+\nFQLB
";

        let mut reader = fastq::io::Reader::new(&data[..]);
        let mut writer = fastq::io::Writer::new(Vec::new());

        let bitmap = BitVec::from_element(0b00000011);

        subsample_exact_single(&mut reader, &mut writer, &bitmap)?;

        let expected = b"@r1\nACGT\n+\nFQLB\n@r2\nACGT\n+\nFQLB\n";
        assert_eq!(writer.get_ref(), expected);

        Ok(())
    }

    #[test]
    fn test_subsample_exact_paired() -> Result<(), SubsampleError> {
        let r1_data = b"@r1\nACGT\n+\nFQLB
@r2\nACGT\n+\nFQLB
@r3\nACGT\n+\nFQLB
@r4\nACGT\n+\nFQLB
";

        let r2_data = b"@r1\nTGCA\n+\nBLQF
@r2\nTGCA\n+\nBLQF
@r3\nTGCA\n+\nBLQF
@r4\nTGCA\n+\nBLQF
";

        let mut r1 = fastq::io::Reader::new(&r1_data[..]);
        let mut w1 = fastq::io::Writer::new(Vec::new());
        let mut r2 = fastq::io::Reader::new(&r2_data[..]);
        let mut w2 = fastq::io::Writer::new(Vec::new());

        let bitmap = BitVec::from_element(0b00000011);

        subsample_exact_paired((&mut r1, &mut w1), (&mut r2, &mut w2), &bitmap)?;

        let w1_expected = b"@r1\nACGT\n+\nFQLB\n@r2\nACGT\n+\nFQLB\n";
        assert_eq!(w1.get_ref(), w1_expected);

        let w2_expected = b"@r1\nTGCA\n+\nBLQF\n@r2\nTGCA\n+\nBLQF\n";
        assert_eq!(w2.get_ref(), w2_expected);

        Ok(())
    }

    #[test]
    fn test_parse_tile_bin() {
        let key = parse_tile_bin(b"@A00226:83:HFWFVDSXX:2:1101:1234:5678");
        assert_eq!(key, Some((2u64 << 32) | 1101));

        let key = parse_tile_bin(b"@A00226:83:HFWFVDSXX:1:2205:9876:4321 1:N:0:ACGTACGT");
        assert_eq!(key, Some((1u64 << 32) | 2205));

        let key = parse_tile_bin(b"@INST:100:FC:4:2301:10:20");
        assert_eq!(key, Some((4u64 << 32) | 2301));

        assert_eq!(parse_tile_bin(b"@INST:100:FC"), None);
        assert_eq!(parse_tile_bin(b"@INST:100:FC:X:1101:1:2"), None);
        assert_eq!(parse_tile_bin(b"@INST:100:FC:1:ABC:1:2"), None);

        let key = parse_tile_bin(b"INST:100:FC:3:1201:1:2");
        assert_eq!(key, Some((3u64 << 32) | 1201));
    }

    #[test]
    fn test_subsample_by_tile_exact_single() -> Result<(), SubsampleError> {
        let dir = std::env::temp_dir().join("fq_test_tile_exact_single");
        std::fs::create_dir_all(&dir).unwrap();

        let r1_data = b"\
@A:1:FC:1:1101:10:20\nACGT\n+\nFFFF
@A:1:FC:1:1101:30:40\nTGCA\n+\nFFFF
@A:1:FC:1:1102:50:60\nGCTA\n+\nFFFF
@A:1:FC:1:1101:70:80\nCGAT\n+\nFFFF
";

        let r1_src = dir.join("r1.fq");
        let r1_dst = dir.join("r1_out.fq");
        std::fs::write(&r1_src, r1_data).unwrap();

        let r1_dsts: Vec<&Path> = vec![&r1_dst];
        let rng = SmallRng::seed_from_u64(42);
        subsample_by_tile(
            (&r1_src, &r1_dsts),
            (None, None),
            rng,
            TileCountMode::Explicit(vec![2]),
            false, // not fast (use exact)
            1,
            1,
            false, // not in-memory
            None,  // default temp dir
        )?;

        let output = std::fs::read_to_string(&r1_dst).unwrap();
        let output_records: Vec<&str> = output.trim().split('\n').collect();
        assert_eq!(
            output_records.len(),
            8,
            "expected 2 records (8 lines), got: {output}"
        );

        for line in output_records.iter().step_by(4) {
            assert!(line.contains(":1101:"), "expected tile 1101, got: {line}");
        }

        std::fs::remove_dir_all(&dir).unwrap();
        Ok(())
    }

    #[test]
    fn test_subsample_by_tile_skip_ahead_single() -> Result<(), SubsampleError> {
        let dir = std::env::temp_dir().join("fq_test_tile_skip_single");
        std::fs::create_dir_all(&dir).unwrap();

        // 10 records on tile 1101, 1 on tile 1102. Target 5 per tile.
        // Tile 1101 retained (10 >= 5), tile 1102 discarded (1 < 5).
        let mut data = Vec::new();
        for i in 0..10 {
            data.extend_from_slice(
                format!("@A:1:FC:1:1101:{}:20\nACGT\n+\nFFFF\n", i * 10).as_bytes(),
            );
        }
        data.extend_from_slice(b"@A:1:FC:1:1102:50:60\nGCTA\n+\nFFFF\n");

        let r1_src = dir.join("r1.fq");
        let r1_dst = dir.join("r1_out.fq");
        std::fs::write(&r1_src, &data).unwrap();

        let r1_dsts: Vec<&Path> = vec![&r1_dst];
        let rng = SmallRng::seed_from_u64(42);
        subsample_by_tile(
            (&r1_src, &r1_dsts),
            (None, None),
            rng,
            TileCountMode::Explicit(vec![5]),
            true, // fast (skip-ahead)
            1,
            1,
            false,
            None,
        )?;

        let output = std::fs::read_to_string(&r1_dst).unwrap();
        let record_count = output.trim().split('\n').count() / 4;
        // Skip-ahead is approximate, so we check it's in a reasonable range
        assert!(
            record_count >= 3 && record_count <= 7,
            "expected ~5 records, got {record_count}"
        );

        // All records should be from tile 1101
        for line in output.trim().split('\n').step_by(4) {
            assert!(line.contains(":1101:"), "expected tile 1101, got: {line}");
        }

        std::fs::remove_dir_all(&dir).unwrap();
        Ok(())
    }

    #[test]
    fn test_subsample_by_tile_exact_paired() -> Result<(), SubsampleError> {
        let dir = std::env::temp_dir().join("fq_test_tile_exact_paired");
        std::fs::create_dir_all(&dir).unwrap();

        let r1_data = b"\
@A:1:FC:1:1101:10:20\nACGT\n+\nFFFF
@A:1:FC:1:1101:30:40\nTGCA\n+\nFFFF
@A:1:FC:1:1102:50:60\nGCTA\n+\nFFFF
@A:1:FC:1:1101:70:80\nCGAT\n+\nFFFF
";
        let r2_data = b"\
@A:1:FC:1:1101:10:20\nAAAA\n+\nFFFF
@A:1:FC:1:1101:30:40\nCCCC\n+\nFFFF
@A:1:FC:1:1102:50:60\nGGGG\n+\nFFFF
@A:1:FC:1:1101:70:80\nTTTT\n+\nFFFF
";

        let r1_src = dir.join("r1.fq");
        let r1_dst = dir.join("r1_out.fq");
        let r2_src = dir.join("r2.fq");
        let r2_dst = dir.join("r2_out.fq");

        std::fs::write(&r1_src, r1_data).unwrap();
        std::fs::write(&r2_src, r2_data).unwrap();

        let r1_dsts: Vec<&Path> = vec![&r1_dst];
        let r2_dsts: Vec<&Path> = vec![&r2_dst];
        let rng = SmallRng::seed_from_u64(42);
        subsample_by_tile(
            (&r1_src, &r1_dsts),
            (Some(r2_src.as_path()), Some(r2_dsts.as_slice())),
            rng,
            TileCountMode::Explicit(vec![2]),
            false, // not fast (use exact)
            1,
            1,
            false,
            None,
        )?;

        let r1_output = std::fs::read_to_string(&r1_dst).unwrap();
        let r2_output = std::fs::read_to_string(&r2_dst).unwrap();

        let r1_lines: Vec<&str> = r1_output.trim().split('\n').collect();
        let r2_lines: Vec<&str> = r2_output.trim().split('\n').collect();

        assert_eq!(r1_lines.len(), 8);
        assert_eq!(r2_lines.len(), 8);

        // R1 and R2 names should match
        for i in (0..r1_lines.len()).step_by(4) {
            let r1_name = r1_lines[i].split(' ').next().unwrap();
            let r2_name = r2_lines[i].split(' ').next().unwrap();
            assert_eq!(r1_name, r2_name);
        }

        std::fs::remove_dir_all(&dir).unwrap();
        Ok(())
    }

    #[test]
    fn test_subsample_by_tile_from_record_count() -> Result<(), SubsampleError> {
        let dir = std::env::temp_dir().join("fq_test_tile_from_count2");
        std::fs::create_dir_all(&dir).unwrap();

        let data = b"\
@A:1:FC:1:1101:10:20\nACGT\n+\nFFFF
@A:1:FC:1:1102:10:20\nTGCA\n+\nFFFF
@A:1:FC:1:1101:30:40\nGCTA\n+\nFFFF
@A:1:FC:1:1102:30:40\nCGAT\n+\nFFFF
@A:1:FC:1:1101:50:60\nACGT\n+\nFFFF
@A:1:FC:1:1102:50:60\nTGCA\n+\nFFFF
";

        let r1_src = dir.join("r1.fq");
        let r1_dst = dir.join("r1_out.fq");
        std::fs::write(&r1_src, data).unwrap();

        let r1_dsts: Vec<&Path> = vec![&r1_dst];
        let rng = SmallRng::seed_from_u64(42);
        subsample_by_tile(
            (&r1_src, &r1_dsts),
            (None, None),
            rng,
            TileCountMode::FromRecordCount(vec![4]),
            false, // not fast (use exact)
            1,
            1,
            false,
            None,
        )?;

        let output = std::fs::read_to_string(&r1_dst).unwrap();
        let output_lines: Vec<&str> = output.trim().split('\n').collect();
        assert_eq!(
            output_lines.len(),
            16,
            "expected 4 records (16 lines), got: {output}"
        );

        std::fs::remove_dir_all(&dir).unwrap();
        Ok(())
    }

    #[test]
    fn test_subsample_by_tile_in_memory() -> Result<(), SubsampleError> {
        let dir = std::env::temp_dir().join("fq_test_tile_inmem");
        std::fs::create_dir_all(&dir).unwrap();

        let r1_data = b"\
@A:1:FC:1:1101:10:20\nACGT\n+\nFFFF
@A:1:FC:1:1101:30:40\nTGCA\n+\nFFFF
@A:1:FC:1:1102:50:60\nGCTA\n+\nFFFF
@A:1:FC:1:1101:70:80\nCGAT\n+\nFFFF
";
        let r2_data = b"\
@A:1:FC:1:1101:10:20\nAAAA\n+\nFFFF
@A:1:FC:1:1101:30:40\nCCCC\n+\nFFFF
@A:1:FC:1:1102:50:60\nGGGG\n+\nFFFF
@A:1:FC:1:1101:70:80\nTTTT\n+\nFFFF
";

        let r1_src = dir.join("r1.fq");
        let r1_dst = dir.join("r1_out.fq");
        let r2_src = dir.join("r2.fq");
        let r2_dst = dir.join("r2_out.fq");

        std::fs::write(&r1_src, r1_data).unwrap();
        std::fs::write(&r2_src, r2_data).unwrap();

        let r1_dsts: Vec<&Path> = vec![&r1_dst];
        let r2_dsts: Vec<&Path> = vec![&r2_dst];
        let rng = SmallRng::seed_from_u64(42);
        subsample_by_tile(
            (&r1_src, &r1_dsts),
            (Some(r2_src.as_path()), Some(r2_dsts.as_slice())),
            rng,
            TileCountMode::Explicit(vec![2]),
            false, // not fast; doesn't matter for in-memory
            1,
            1,
            true, // in-memory
            None,
        )?;

        let r1_output = std::fs::read_to_string(&r1_dst).unwrap();
        let r2_output = std::fs::read_to_string(&r2_dst).unwrap();

        let r1_lines: Vec<&str> = r1_output.trim().split('\n').collect();
        let r2_lines: Vec<&str> = r2_output.trim().split('\n').collect();

        // 2 records from tile 1101 (3 records, 2 selected), tile 1102 discarded (1 < 2)
        assert_eq!(r1_lines.len(), 8, "expected 2 R1 records, got: {r1_output}");
        assert_eq!(r2_lines.len(), 8, "expected 2 R2 records, got: {r2_output}");

        // All R1 records from tile 1101
        for line in r1_lines.iter().step_by(4) {
            assert!(line.contains(":1101:"), "expected tile 1101, got: {line}");
        }

        std::fs::remove_dir_all(&dir).unwrap();
        Ok(())
    }

    #[test]
    fn test_subsample_exact_multi_rate_nested() -> Result<(), SubsampleError> {
        let dir = std::env::temp_dir().join("fq_test_exact_multi");
        std::fs::create_dir_all(&dir).unwrap();

        // 20 records, ask for both 5 and 10 — smaller output must be a subset.
        let mut data = Vec::new();
        for i in 0..20 {
            data.extend_from_slice(format!("@r{i}\nACGT\n+\nFFFF\n").as_bytes());
        }
        let src = dir.join("r1.fq");
        let dst_small = dir.join("r1_n5.fq");
        let dst_large = dir.join("r1_n10.fq");
        std::fs::write(&src, &data).unwrap();

        let rng = SmallRng::seed_from_u64(42);
        let r1_dsts: Vec<&Path> = vec![&dst_small, &dst_large];
        subsample_exact((&src, &r1_dsts), (None, None), rng, &[5, 10], false)?;

        let read_names = |p: &Path| -> Vec<String> {
            let s = std::fs::read_to_string(p).unwrap();
            s.lines().step_by(4).map(|l| l.to_string()).collect()
        };

        let small = read_names(&dst_small);
        let large = read_names(&dst_large);
        assert_eq!(small.len(), 5, "expected 5 records in small output");
        assert_eq!(large.len(), 10, "expected 10 records in large output");
        // Every small record must be in the large output.
        let large_set: std::collections::HashSet<_> = large.iter().cloned().collect();
        for name in &small {
            assert!(
                large_set.contains(name),
                "small record {name} not in large output"
            );
        }

        std::fs::remove_dir_all(&dir).unwrap();
        Ok(())
    }

    #[test]
    fn test_build_nested_filters_nests() -> Result<(), SubsampleError> {
        let rng = SmallRng::seed_from_u64(7);
        // Request 3, 5, 10 from 20 records — largest bitmap gets 10 set bits,
        // and each smaller is a subset of the next larger.
        let counts = [3u64, 5, 10];
        let bitmaps = build_nested_filters(rng, 20, &counts)?;
        assert_eq!(bitmaps.len(), 3);
        for (i, bm) in bitmaps.iter().enumerate() {
            let ones = bm.iter_ones().count() as u64;
            assert_eq!(ones, counts[i], "bitmap {i} has wrong count");
        }
        // Each smaller is a subset of the next larger.
        for i in 0..bitmaps.len() - 1 {
            for idx in bitmaps[i].iter_ones() {
                assert!(
                    bitmaps[i + 1][idx],
                    "bitmap {i} is not a subset of {}",
                    i + 1
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_format_count_label() {
        assert_eq!(format_count_label('n', 500), "n500");
        assert_eq!(format_count_label('n', 10_000), "n10K");
        assert_eq!(
            format_count_label('t', 2_500),
            "t2_500".to_string().replace('_', "")
        );
        assert_eq!(format_count_label('n', 2_000_000), "n2M");
        assert_eq!(format_count_label('t', 30_000), "t30K");
        assert_eq!(format_count_label('n', 0), "n0");
        // Non-round values fall through to the raw number.
        assert_eq!(format_count_label('n', 1_234), "n1234");
    }

    #[test]
    fn test_format_probability_label() {
        assert_eq!(format_probability_label(0.05), "p05");
        assert_eq!(format_probability_label(0.5), "p50");
        assert_eq!(format_probability_label(0.99), "p99");
    }

    #[test]
    fn test_expand_template() {
        assert_eq!(
            expand_template("out/r1_{quantity}.fq.gz", "p05", "0.05"),
            "out/r1_p05.fq.gz"
        );
        assert_eq!(
            expand_template("r1_{value}.fq", "n10K", "10000"),
            "r1_10000.fq"
        );
    }

    #[test]
    fn test_subsample_by_tile_all_bins_too_small() -> Result<(), SubsampleError> {
        let dir = std::env::temp_dir().join("fq_test_tile_empty2");
        std::fs::create_dir_all(&dir).unwrap();

        let data = b"\
@A:1:FC:1:1101:10:20\nACGT\n+\nFFFF
@A:1:FC:1:1101:30:40\nTGCA\n+\nFFFF
@A:1:FC:1:1102:50:60\nGCTA\n+\nFFFF
@A:1:FC:1:1102:70:80\nCGAT\n+\nFFFF
";

        let r1_src = dir.join("r1.fq");
        let r1_dst = dir.join("r1_out.fq");
        std::fs::write(&r1_src, data).unwrap();

        let r1_dsts: Vec<&Path> = vec![&r1_dst];
        let rng = SmallRng::seed_from_u64(42);
        subsample_by_tile(
            (&r1_src, &r1_dsts),
            (None, None),
            rng,
            TileCountMode::Explicit(vec![3]),
            false,
            1,
            1,
            false,
            None,
        )?;

        let output = std::fs::read_to_string(&r1_dst).unwrap();
        assert!(output.is_empty(), "expected empty output, got: {output}");

        std::fs::remove_dir_all(&dir).unwrap();
        Ok(())
    }

    // ---- with-replacement / oversampling ----

    fn data_5() -> &'static [u8] {
        b"@r1\nAAAA\n+\nFFFF\n@r2\nCCCC\n+\nFFFF\n@r3\nGGGG\n+\nFFFF\n@r4\nTTTT\n+\nFFFF\n@r5\nACGT\n+\nFFFF\n"
    }

    fn count_records(buf: &[u8]) -> usize {
        // Each record is 4 lines; count name lines (start with '@' at a line boundary).
        std::str::from_utf8(buf)
            .unwrap()
            .lines()
            .step_by(4)
            .filter(|l| l.starts_with('@'))
            .count()
    }

    #[test]
    fn test_build_with_replacement_sums_to_count_and_repeats() -> Result<(), SubsampleError> {
        // Drawing 20 indices from 5 records with replacement must yield total
        // multiplicity 20 and force at least one record's multiplicity above 1.
        let rng = SmallRng::seed_from_u64(7);
        let mults = build_with_replacement(rng, 5, &[20])?;
        assert_eq!(mults.len(), 1);
        let total: u64 = mults[0].iter().sum();
        assert_eq!(total, 20);
        assert!(
            mults[0].iter().any(|&m| m > 1),
            "expected a repeated record"
        );
        Ok(())
    }

    #[test]
    fn test_exact_with_replacement_emits_exact_count() -> Result<(), SubsampleError> {
        let dir = std::env::temp_dir().join("fq_test_exact_repl");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("in.fq");
        let dst = dir.join("out.fq");
        std::fs::write(&src, data_5()).unwrap();

        let r1_dsts: Vec<&Path> = vec![dst.as_path()];
        let rng = SmallRng::seed_from_u64(7);
        // -n 5 with replacement: exactly 5 records out of a 5-record input.
        subsample_exact((&src, &r1_dsts), (None, None), rng, &[5], true)?;

        let out = std::fs::read(&dst).unwrap();
        assert_eq!(count_records(&out), 5);

        // Oversample: -n 8 with replacement exceeds the input size on purpose.
        let dst2 = dir.join("out8.fq");
        let r1_dsts2: Vec<&Path> = vec![dst2.as_path()];
        let rng2 = SmallRng::seed_from_u64(7);
        subsample_exact((&src, &r1_dsts2), (None, None), rng2, &[8], true)?;
        let out2 = std::fs::read(&dst2).unwrap();
        assert_eq!(count_records(&out2), 8, "oversample should emit exactly 8");

        std::fs::remove_dir_all(&dir).unwrap();
        Ok(())
    }

    #[test]
    fn test_exact_with_replacement_reproducible() -> Result<(), SubsampleError> {
        let dir = std::env::temp_dir().join("fq_test_exact_repl_repro");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("in.fq");
        std::fs::write(&src, data_5()).unwrap();

        let run = |name: &str| -> Vec<u8> {
            let dst = dir.join(name);
            let r1_dsts: Vec<&Path> = vec![dst.as_path()];
            let rng = SmallRng::seed_from_u64(99);
            subsample_exact((&src, &r1_dsts), (None, None), rng, &[7], true).unwrap();
            std::fs::read(&dst).unwrap()
        };
        assert_eq!(run("a.fq"), run("b.fq"), "same seed must reproduce output");

        std::fs::remove_dir_all(&dir).unwrap();
        Ok(())
    }

    #[test]
    fn test_streaming_with_replacement_multiplicities() -> Result<(), SubsampleError> {
        // Poisson(1.0) per record can emit a record more than once.
        let mut reader = fastq::io::Reader::new(data_5());
        let mut writers = vec![fastq::io::Writer::new(Vec::new())];
        let mut rng = SmallRng::seed_from_u64(3);

        let (counts, total) =
            subsample_single_multi(&mut reader, &mut writers, &mut rng, &[1.0], true)?;
        assert_eq!(total, 5);
        // Emitted count equals the sum of multiplicities reported.
        assert_eq!(counts[0], count_records(writers[0].get_ref()) as u64);
        Ok(())
    }

    #[test]
    fn test_streaming_oversample_fraction_gt_one() -> Result<(), SubsampleError> {
        // fraction 3.0 with replacement: a 5-record input can emit > 5 records.
        let mut reader = fastq::io::Reader::new(data_5());
        let mut writers = vec![fastq::io::Writer::new(Vec::new())];
        let mut rng = SmallRng::seed_from_u64(11);
        let (counts, _) =
            subsample_single_multi(&mut reader, &mut writers, &mut rng, &[3.0], true)?;
        assert!(
            counts[0] > 5,
            "fraction 3.0 should oversample, got {}",
            counts[0]
        );
        Ok(())
    }

    #[test]
    fn test_non_finite_fraction_rejected_not_panic() {
        // Regression: a non-finite fraction must be rejected up front by the
        // resolver rather than flowing into Poisson::new and panicking.
        for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let mut args = base_args();
            args.fraction = vec![bad];
            args.with_replacement = true;
            assert!(
                matches!(
                    resolve_quantity(&args),
                    Err(SubsampleError::InvalidFraction(_))
                ),
                "resolve_quantity accepted {bad}"
            );
        }
        // A finite-but-too-large lambda passes the resolver's finiteness check
        // (it is finite and > 0.0) but must still error from poisson_per_rate
        // rather than panicking via .expect().
        assert!(
            poisson_per_rate(&[1e30], true).is_err(),
            "poisson_per_rate accepted an over-large lambda"
        );
        // Non-finite values are rejected by poisson_per_rate too (defense in depth).
        assert!(poisson_per_rate(&[f64::INFINITY], true).is_err());
        assert!(poisson_per_rate(&[f64::NAN], true).is_err());
    }

    #[test]
    fn test_resolve_quantity_fraction_gt_one_requires_replacement() {
        let mut args = base_args();
        args.fraction = vec![2.0];
        args.with_replacement = false;
        assert!(matches!(
            resolve_quantity(&args),
            Err(SubsampleError::FractionRequiresReplacement(_))
        ));

        args.with_replacement = true;
        assert!(resolve_quantity(&args).is_ok());
    }

    #[test]
    fn test_resolve_quantity_probability_alias_still_parses() {
        // The hidden --probability alias populates the same field, so a value in
        // (0, 1) without replacement must resolve to a Fraction quantity.
        let mut args = base_args();
        args.fraction = vec![0.5];
        match resolve_quantity(&args).unwrap() {
            Quantity::Fraction(v) => assert_eq!(v, vec![0.5]),
            other => panic!("expected Fraction, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_destinations_single_rate_template_needs_no_token() -> Result<(), SubsampleError>
    {
        // A single-rate template may omit {quantity}/{value} (e.g. when only
        // {sample} drives the path, already expanded by the time we get here).
        let q = Quantity::RecordCount(vec![3]);
        let dsts = resolve_destinations(None, Some("/tmp/out_s1.fq"), &q, "r1")?;
        assert_eq!(dsts, vec![PathBuf::from("/tmp/out_s1.fq")]);

        // But a multi-rate template without a token is still rejected.
        let q2 = Quantity::RecordCount(vec![2, 3]);
        assert!(matches!(
            resolve_destinations(None, Some("/tmp/out.fq"), &q2, "r1"),
            Err(SubsampleError::TemplateMissingToken(_))
        ));
        Ok(())
    }

    #[test]
    fn test_expand_sample() {
        // Single replicate: unchanged.
        assert_eq!(
            expand_sample("out/r1_{quantity}.fq.gz", 0, 1),
            "out/r1_{quantity}.fq.gz"
        );
        // Explicit {sample} token.
        assert_eq!(expand_sample("out/r1.{sample}.fq", 1, 3), "out/r1.s2.fq");
        // No token: insert `.sN` before the first extension separator.
        assert_eq!(
            expand_sample("out/r1_p50.fq.gz", 0, 2),
            "out/r1_p50.s1.fq.gz"
        );
        // No extension at all.
        assert_eq!(expand_sample("out_r1", 2, 4), "out_r1.s3");
    }

    /// A minimal `SubsampleArgs` for unit-testing the pure resolver helpers.
    fn base_args() -> SubsampleArgs {
        SubsampleArgs {
            fraction: Vec::new(),
            record_count: Vec::new(),
            record_count_per_tile: Vec::new(),
            bin_by_tile: false,
            fast: false,
            with_replacement: false,
            num_samples: 1,
            in_memory: false,
            temp_dir: None,
            sampling_threads: 1,
            compression_threads: 1,
            seed: Some(0),
            r1_dst: None,
            r2_dst: None,
            r1_dst_template: None,
            r2_dst_template: None,
            r1_src: PathBuf::from("in.fq"),
            r2_src: None,
        }
    }
}
