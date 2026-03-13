use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, BufRead, BufReader, Write},
    ops::{Bound, RangeBounds},
    path::{Path, PathBuf},
};

use bitvec::vec::BitVec;
use flate2::bufread::MultiGzDecoder;
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
    rngs::SmallRng,
};
use thiserror::Error;
use tracing::{info, info_span, warn};

use crate::{
    cli::SubsampleArgs,
    fastq::{self, Record},
};

const VALID_PROBABILITY_RANGE: (Bound<f64>, Bound<f64>) =
    (Bound::Excluded(0.0), Bound::Excluded(1.0));

pub fn subsample(args: SubsampleArgs) -> Result<(), SubsampleError> {
    let r1_src = &args.r1_src;
    let r1_dst = &args.r1_dst;

    let r2_src = args.r2_src.as_ref();
    let r2_dst = args.r2_dst.as_ref();

    info!(command = "subsample", "fq");

    let rng = if let Some(seed) = args.seed {
        info!(seed = seed, "initializing rng from seed");
        SmallRng::seed_from_u64(seed)
    } else {
        info!("initializing rng from entropy");
        SmallRng::from_os_rng()
    };

    if let Some(probability) = args.probability {
        subsample_approximate(
            (r1_src, r1_dst),
            (r2_src.map(|p| &**p), r2_dst.map(|p| &**p)),
            rng,
            probability,
        )?;
    } else if let Some(record_count) = args.record_count {
        subsample_exact(
            (r1_src, r1_dst),
            (r2_src.map(|p| &**p), r2_dst.map(|p| &**p)),
            rng,
            record_count,
        )?;
    } else if let Some(record_count_per_tile) = args.record_count_per_tile {
        subsample_by_tile(
            (r1_src, r1_dst),
            (r2_src.map(|p| &**p), r2_dst.map(|p| &**p)),
            rng,
            record_count_per_tile,
        )?;
    } else {
        unreachable!();
    }

    info!("done");

    Ok(())
}

fn subsample_approximate<Rng>(
    (r1_src, r1_dst): (&Path, &Path),
    (r2_src, r2_dst): (Option<&Path>, Option<&Path>),
    mut rng: Rng,
    probability: f64,
) -> Result<(), SubsampleError>
where
    Rng: rand::Rng,
{
    if !VALID_PROBABILITY_RANGE.contains(&probability) {
        return Err(SubsampleError::InvalidProbability(probability));
    }

    let mut r1 = fastq::fs::open(r1_src).map_err(|e| SubsampleError::OpenFile(e, r1_src.into()))?;
    let mut w1 =
        fastq::fs::create(r1_dst).map_err(|e| SubsampleError::CreateFile(e, r1_dst.into()))?;

    let span = info_span!("subsample_approximate", probability = probability);
    let _span_ctx = span.enter();

    let (n, total) = match (r2_src, r2_dst) {
        (Some(r2_src), Some(r2_dst)) => {
            info!("sampling paired end reads");

            let mut r2 =
                fastq::fs::open(r2_src).map_err(|e| SubsampleError::OpenFile(e, r2_src.into()))?;
            let mut w2 = fastq::fs::create(r2_dst)
                .map_err(|e| SubsampleError::CreateFile(e, r2_dst.into()))?;

            subsample_paired(
                (&mut r1, &mut w1),
                (&mut r2, &mut w2),
                &mut rng,
                probability,
            )?
        }
        (Some(_), None) => return Err(SubsampleError::MissingDestination("r2-dst")),
        (None, Some(_)) => return Err(SubsampleError::MissingSource("r2-src")),
        _ => {
            info!("sampling single end reads");
            subsample_single(&mut r1, &mut w1, &mut rng, probability)?
        }
    };

    let percentage = (n as f64) / (total as f64) * 100.0;
    info!("sampled {}/{} ({:.1}%) records", n, total, percentage);

    Ok(())
}

fn subsample_single<R, W, Rng>(
    reader: &mut fastq::io::Reader<R>,
    writer: &mut fastq::io::Writer<W>,
    rng: &mut Rng,
    p: f64,
) -> Result<(u64, u64), SubsampleError>
where
    R: BufRead,
    W: Write,
    Rng: rand::Rng,
{
    let mut record = Record::default();

    let mut n = 0;
    let mut total = 0;

    while reader.read_record(&mut record)? != 0 {
        let q: f64 = rng.random();

        if q <= p {
            writer.write_record(&record)?;
            n += 1;
        }

        total += 1;
    }

    Ok((n, total))
}

fn subsample_paired<R, S, W, X, Rng>(
    (r1, w1): (&mut fastq::io::Reader<R>, &mut fastq::io::Writer<W>),
    (r2, w2): (&mut fastq::io::Reader<S>, &mut fastq::io::Writer<X>),
    rng: &mut Rng,
    p: f64,
) -> Result<(u64, u64), SubsampleError>
where
    R: BufRead,
    S: BufRead,
    W: Write,
    X: Write,
    Rng: rand::Rng,
{
    let mut s1 = Record::default();
    let mut s2 = Record::default();

    let mut n = 0;
    let mut total = 0;

    loop {
        match (r1.read_record(&mut s1)?, r2.read_record(&mut s2)?) {
            (0, 0) => break,
            (0, len) if len > 0 => return Err(SubsampleError::UnexpectedEof("r1-src")),
            (len, 0) if len > 0 => return Err(SubsampleError::UnexpectedEof("r2-src")),
            (_, _) => {
                let q: f64 = rng.random();

                if q <= p {
                    w1.write_record(&s1)?;
                    w2.write_record(&s2)?;
                    n += 1;
                }

                total += 1;
            }
        }
    }

    Ok((n, total))
}

fn subsample_exact<Rng>(
    (r1_src, r1_dst): (&Path, &Path),
    (r2_src, r2_dst): (Option<&Path>, Option<&Path>),
    rng: Rng,
    mut record_count: u64,
) -> Result<(), SubsampleError>
where
    Rng: rand::Rng,
{
    let span = info_span!("subsample_exact", record_count = record_count);
    let _span_ctx = span.enter();

    info!("counting records");

    let line_count = count_lines(r1_src)?;
    let actual_record_count = line_count / 4;

    info!(actual_record_count = actual_record_count, "counted records");

    let n = u64::try_from(actual_record_count)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    if record_count > n {
        warn!(
            "record count ({}) > r1-src record count ({}). Using record-count = {} instead.",
            record_count, n, n
        );

        record_count = n;
    }

    info!("building filter");
    let bitmap = build_filter(rng, actual_record_count, record_count)?;
    info!("built filter");

    let mut r1 = fastq::fs::open(r1_src).map_err(|e| SubsampleError::OpenFile(e, r1_src.into()))?;
    let mut w1 =
        fastq::fs::create(r1_dst).map_err(|e| SubsampleError::CreateFile(e, r1_dst.into()))?;

    match (r2_src, r2_dst) {
        (Some(r2_src), Some(r2_dst)) => {
            info!("sampling paired end reads");

            let mut r2 =
                fastq::fs::open(r2_src).map_err(|e| SubsampleError::OpenFile(e, r2_src.into()))?;
            let mut w2 = fastq::fs::create(r2_dst)
                .map_err(|e| SubsampleError::CreateFile(e, r2_dst.into()))?;

            subsample_exact_paired((&mut r1, &mut w1), (&mut r2, &mut w2), &bitmap)?;
        }
        (Some(_), None) => return Err(SubsampleError::MissingDestination("r2-dst")),
        (None, Some(_)) => return Err(SubsampleError::MissingSource("r2-src")),
        (None, None) => {
            info!("sampling single end reads");
            subsample_exact_single(&mut r1, &mut w1, &bitmap)?;
        }
    }

    let percentage = (record_count as f64) / (actual_record_count as f64) * 100.0;
    info!(
        "sampled {}/{} ({:.1}%) records",
        record_count, actual_record_count, percentage
    );

    Ok(())
}

fn count_lines<P>(src: P) -> io::Result<usize>
where
    P: AsRef<Path>,
{
    const LINE_FEED: u8 = b'\n';

    let mut reader = open(src)?;
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

fn open<P>(src: P) -> io::Result<Box<dyn BufRead>>
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

fn random_sample<Rng>(rng: &mut Rng, population: usize, n: usize) -> HashSet<usize>
where
    Rng: rand::Rng,
{
    let distribution = Uniform::new(0, population).unwrap();
    let mut selected = HashSet::with_capacity(n);

    while selected.len() < n {
        let i = distribution.sample(rng);
        selected.insert(i);
    }

    selected
}

fn subsample_by_tile<Rng>(
    (r1_src, r1_dst): (&Path, &Path),
    (r2_src, r2_dst): (Option<&Path>, Option<&Path>),
    mut rng: Rng,
    record_count_per_tile: u64,
) -> Result<(), SubsampleError>
where
    Rng: rand::Rng,
{
    let span = info_span!("subsample_by_tile", record_count_per_tile);
    let _span_ctx = span.enter();

    // First pass: read R1 to bin records by (lane, tile)
    info!("first pass: binning records by tile");

    let mut r1 = fastq::fs::open(r1_src).map_err(|e| SubsampleError::OpenFile(e, r1_src.into()))?;
    let mut record = Record::default();

    let mut bin_assignments: Vec<u64> = Vec::new();
    let mut bin_counts: HashMap<u64, usize> = HashMap::new();
    let mut parse_failures: u64 = 0;

    while r1.read_record(&mut record)? != 0 {
        if let Some(bin_key) = parse_tile_bin(record.name()) {
            bin_assignments.push(bin_key);
            *bin_counts.entry(bin_key).or_insert(0) += 1;
        } else {
            bin_assignments.push(u64::MAX); // sentinel for unparseable
            parse_failures += 1;
        }
    }

    let total_records = bin_assignments.len();
    info!(
        total_records,
        parse_failures,
        bins = bin_counts.len(),
        "binning complete"
    );

    if parse_failures > 0 {
        warn!(
            "{} records had headers that could not be parsed as Illumina format",
            parse_failures
        );
    }

    // For each retained bin, randomly select record_count_per_tile positions
    let n = record_count_per_tile as usize;
    let mut retained_bins: HashMap<u64, HashSet<usize>> = HashMap::new();

    for (&bin_key, &count) in &bin_counts {
        if count >= n {
            let sample = random_sample(&mut rng, count, n);
            retained_bins.insert(bin_key, sample);
        }
    }

    let retained_bin_count = retained_bins.len();
    let discarded_bin_count = bin_counts.len() - retained_bin_count;
    info!(
        retained_bins = retained_bin_count,
        discarded_bins = discarded_bin_count,
        "filtered bins"
    );

    if retained_bin_count == 0 {
        warn!("no bins have enough records; output will be empty");
    }

    // Build global bitmap
    info!("building selection bitmap");
    let mut bitmap = BitVec::new();
    bitmap.resize(total_records, false);

    let mut bin_positions: HashMap<u64, usize> = HashMap::new();

    for (i, &bin_key) in bin_assignments.iter().enumerate() {
        if let Some(selected) = retained_bins.get(&bin_key) {
            let pos = bin_positions.entry(bin_key).or_insert(0);
            if selected.contains(pos) {
                bitmap.set(i, true);
            }
            *pos += 1;
        }
    }

    drop(bin_assignments);
    drop(retained_bins);
    drop(bin_positions);

    let selected_count = bitmap.count_ones();
    info!(selected_records = selected_count, "built selection bitmap");

    // Second pass: write selected records
    info!("second pass: writing selected records");

    let mut r1 = fastq::fs::open(r1_src).map_err(|e| SubsampleError::OpenFile(e, r1_src.into()))?;
    let mut w1 =
        fastq::fs::create(r1_dst).map_err(|e| SubsampleError::CreateFile(e, r1_dst.into()))?;

    match (r2_src, r2_dst) {
        (Some(r2_src), Some(r2_dst)) => {
            info!("sampling paired end reads");

            let mut r2 =
                fastq::fs::open(r2_src).map_err(|e| SubsampleError::OpenFile(e, r2_src.into()))?;
            let mut w2 = fastq::fs::create(r2_dst)
                .map_err(|e| SubsampleError::CreateFile(e, r2_dst.into()))?;

            subsample_exact_paired((&mut r1, &mut w1), (&mut r2, &mut w2), &bitmap)?;
        }
        (Some(_), None) => return Err(SubsampleError::MissingDestination("r2-dst")),
        (None, Some(_)) => return Err(SubsampleError::MissingSource("r2-src")),
        (None, None) => {
            info!("sampling single end reads");
            subsample_exact_single(&mut r1, &mut w1, &bitmap)?;
        }
    }

    let percentage = if total_records > 0 {
        (selected_count as f64) / (total_records as f64) * 100.0
    } else {
        0.0
    };
    info!(
        "sampled {}/{} ({:.1}%) records across {} tiles",
        selected_count, total_records, percentage, retained_bin_count
    );

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
    #[error("missing pair source: {0}")]
    MissingSource(&'static str),
    #[error("missing pair destination: {0}")]
    MissingDestination(&'static str),
    #[error("invalid probability: expected (0.0, 1.0), got {0}")]
    InvalidProbability(f64),
    #[error("{0} unexpectedly ended")]
    UnexpectedEof(&'static str),
    #[error("invalid uniform range")]
    InvalidUniformRange(rand::distr::uniform::Error),
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
        let mut writer = fastq::io::Writer::new(Vec::new());

        let mut rng = SmallRng::seed_from_u64(0);

        subsample_single(&mut reader, &mut writer, &mut rng, 0.33)?;

        let expected = b"@r1\nACGT\n+\nFQLB\n@r4\nACGT\n+\nFQLB\n";
        assert_eq!(writer.get_ref(), expected);

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
        let mut w1 = fastq::io::Writer::new(Vec::new());
        let mut r2 = fastq::io::Reader::new(&r2_data[..]);
        let mut w2 = fastq::io::Writer::new(Vec::new());

        let mut rng = SmallRng::seed_from_u64(0);

        subsample_paired((&mut r1, &mut w1), (&mut r2, &mut w2), &mut rng, 0.33)?;

        let w1_expected = b"@r1\nACGT\n+\nFQLB\n@r4\nACGT\n+\nFQLB\n";
        assert_eq!(w1.get_ref(), w1_expected);

        let w2_expected = b"@r1\nTGCA\n+\nBLQF\n@r4\nTGCA\n+\nBLQF\n";
        assert_eq!(w2.get_ref(), w2_expected);

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
        // Standard Illumina format
        let key = parse_tile_bin(b"@A00226:83:HFWFVDSXX:2:1101:1234:5678");
        assert_eq!(key, Some((2u64 << 32) | 1101));

        // With description
        let key = parse_tile_bin(b"@A00226:83:HFWFVDSXX:1:2205:9876:4321 1:N:0:ACGTACGT");
        assert_eq!(key, Some((1u64 << 32) | 2205));

        // Different lane/tile values
        let key = parse_tile_bin(b"@INST:100:FC:4:2301:10:20");
        assert_eq!(key, Some((4u64 << 32) | 2301));

        // Too few fields
        assert_eq!(parse_tile_bin(b"@INST:100:FC"), None);

        // Non-numeric lane
        assert_eq!(parse_tile_bin(b"@INST:100:FC:X:1101:1:2"), None);

        // Non-numeric tile
        assert_eq!(parse_tile_bin(b"@INST:100:FC:1:ABC:1:2"), None);

        // No @ prefix still works
        let key = parse_tile_bin(b"INST:100:FC:3:1201:1:2");
        assert_eq!(key, Some((3u64 << 32) | 1201));
    }

    #[test]
    fn test_random_sample() {
        let mut rng = SmallRng::seed_from_u64(42);

        let sample = random_sample(&mut rng, 100, 10);
        assert_eq!(sample.len(), 10);
        assert!(sample.iter().all(|&i| i < 100));

        // All elements unique
        let as_vec: Vec<_> = sample.iter().collect();
        let as_set: HashSet<_> = as_vec.iter().collect();
        assert_eq!(as_vec.len(), as_set.len());
    }

    #[test]
    fn test_subsample_by_tile_single() -> Result<(), SubsampleError> {
        use std::io::Write;

        let dir = std::env::temp_dir().join("fq_test_tile_single");
        std::fs::create_dir_all(&dir).unwrap();

        // 4 records: 3 on tile 1101, 1 on tile 1102. With n=2, tile 1101 is retained,
        // tile 1102 is discarded. 2 of the 3 records from tile 1101 are selected.
        let r1_data = b"\
@A:1:FC:1:1101:10:20\nACGT\n+\nFFFF
@A:1:FC:1:1101:30:40\nTGCA\n+\nFFFF
@A:1:FC:1:1102:50:60\nGCTA\n+\nFFFF
@A:1:FC:1:1101:70:80\nCGAT\n+\nFFFF
";

        let r1_src = dir.join("r1.fq");
        let r1_dst = dir.join("r1_out.fq");

        {
            let mut f = File::create(&r1_src).unwrap();
            f.write_all(r1_data).unwrap();
        }

        let rng = SmallRng::seed_from_u64(42);
        subsample_by_tile(
            (&r1_src, &r1_dst),
            (None, None),
            rng,
            2,
        )?;

        let output = std::fs::read_to_string(&r1_dst).unwrap();
        let output_records: Vec<&str> = output.trim().split('\n').collect();
        // 2 records selected = 8 lines (4 lines per FASTQ record)
        assert_eq!(output_records.len(), 8, "expected 2 records (8 lines), got: {output}");

        // All selected records should be from tile 1101
        for line in output_records.iter().step_by(4) {
            assert!(line.contains(":1101:"), "expected tile 1101, got: {line}");
        }

        std::fs::remove_dir_all(&dir).unwrap();

        Ok(())
    }

    #[test]
    fn test_subsample_by_tile_paired() -> Result<(), SubsampleError> {
        use std::io::Write;

        let dir = std::env::temp_dir().join("fq_test_tile_paired");
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

        {
            let mut f = File::create(&r1_src).unwrap();
            f.write_all(r1_data).unwrap();
            let mut f = File::create(&r2_src).unwrap();
            f.write_all(r2_data).unwrap();
        }

        let rng = SmallRng::seed_from_u64(42);
        subsample_by_tile(
            (&r1_src, &r1_dst),
            (Some(r2_src.as_path()), Some(r2_dst.as_path())),
            rng,
            2,
        )?;

        let r1_output = std::fs::read_to_string(&r1_dst).unwrap();
        let r2_output = std::fs::read_to_string(&r2_dst).unwrap();

        let r1_lines: Vec<&str> = r1_output.trim().split('\n').collect();
        let r2_lines: Vec<&str> = r2_output.trim().split('\n').collect();

        // Both should have same number of records
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
    fn test_subsample_by_tile_all_bins_too_small() -> Result<(), SubsampleError> {
        use std::io::Write;

        let dir = std::env::temp_dir().join("fq_test_tile_empty");
        std::fs::create_dir_all(&dir).unwrap();

        // Only 2 records per tile, require 3
        let data = b"\
@A:1:FC:1:1101:10:20\nACGT\n+\nFFFF
@A:1:FC:1:1101:30:40\nTGCA\n+\nFFFF
@A:1:FC:1:1102:50:60\nGCTA\n+\nFFFF
@A:1:FC:1:1102:70:80\nCGAT\n+\nFFFF
";

        let r1_src = dir.join("r1.fq");
        let r1_dst = dir.join("r1_out.fq");

        {
            let mut f = File::create(&r1_src).unwrap();
            f.write_all(data).unwrap();
        }

        let rng = SmallRng::seed_from_u64(42);
        subsample_by_tile((&r1_src, &r1_dst), (None, None), rng, 3)?;

        let output = std::fs::read_to_string(&r1_dst).unwrap();
        assert!(output.is_empty(), "expected empty output, got: {output}");

        std::fs::remove_dir_all(&dir).unwrap();

        Ok(())
    }
}
