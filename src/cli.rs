use std::{path::PathBuf, str::FromStr};

use clap::{ArgGroup, Parser, Subcommand};
use git_testament::{git_testament, render_testament};
use regex::bytes::Regex;

use crate::{ValidationLevel, validators::LintMode};

git_testament!(TESTAMENT);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AsciiChar(u8);

impl From<AsciiChar> for u8 {
    fn from(c: AsciiChar) -> Self {
        c.0
    }
}

impl FromStr for AsciiChar {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let [b] = s.as_bytes()
            && b.is_ascii()
        {
            Ok(Self(*b))
        } else {
            Err("invalid ASCII character")
        }
    }
}

#[derive(Parser)]
#[command(propagate_version = true, version = render_testament!(TESTAMENT))]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Collect FASTQ metrics.
    Describe(DescribeArgs),
    /// Filters a FASTQ file.
    Filter(FilterArgs),
    /// Validates a FASTQ file pair.
    Lint(LintArgs),
    /// Outputs a subset of records.
    Subsample(SubsampleArgs),
}

#[derive(Parser)]
pub struct DescribeArgs {
    /// FASTQ source.
    pub src: PathBuf,
}

#[derive(Parser)]
#[command(group(ArgGroup::new("filter").required(true).args(["names", "sequence_pattern"])))]
pub struct FilterArgs {
    /// Allowlist of record names.
    #[arg(long)]
    pub names: Option<PathBuf>,

    /// Keep records that have sequences that match the given regular expression.
    #[arg(long)]
    pub sequence_pattern: Option<Regex>,

    /// Filtered FASTQ destinations.
    #[arg(long, required = true)]
    pub dsts: Vec<PathBuf>,

    /// FASTQ sources. Accepts both raw and gzipped FASTQ inputs.
    pub srcs: Vec<PathBuf>,
}

#[derive(Parser)]
pub struct LintArgs {
    /// Panic on first error or log all errors.
    #[arg(long, value_enum, default_value_t = LintMode::Panic)]
    pub lint_mode: LintMode,

    /// Only use single read validators up to a given level.
    #[arg(long, value_enum, default_value_t = ValidationLevel::High)]
    pub single_read_validation_level: ValidationLevel,

    /// Only use paired read validators up to a given level.
    #[arg(long, value_enum, default_value_t = ValidationLevel::High)]
    pub paired_read_validation_level: ValidationLevel,

    /// Disable validators by code. Use multiple times to disable more than one.
    #[arg(long)]
    pub disable_validator: Vec<String>,

    /// Define a record definition separator.
    ///
    /// This is used to strip the description from a record name.
    ///
    /// [default: '/' and ' ']
    #[arg(long)]
    pub record_definition_separator: Option<AsciiChar>,

    /// Read 1 source. Accepts both raw and gzipped FASTQ inputs.
    pub r1_src: PathBuf,

    /// Read 2 source. Accepts both raw and gzipped FASTQ inputs.
    pub r2_src: Option<PathBuf>,
}

#[derive(Parser)]
#[command(group(
    ArgGroup::new("quantity")
        .required(true)
        .multiple(true)
        .args(["fraction", "record_count", "record_count_per_tile"])
))]
pub struct SubsampleArgs {
    /// The fraction of records to keep. Without --with-replacement this must lie in (0.0, 1.0)
    /// and selects each record independently (Bernoulli). With --with-replacement it may be any
    /// value > 0.0: a fraction > 1.0 oversamples (e.g. 2.0 emits ~2x the input). May be given
    /// more than once to emit one output per rate (requires --r1-dst-template); without
    /// replacement the smaller sets are subsets of the larger ones.
    /// Cannot be used with `record-count` or `record-count-per-tile`.
    #[arg(short = 'p', long, alias = "probability", num_args = 1..)]
    pub fraction: Vec<f64>,

    /// The exact number of records to keep. Without --with-replacement these are drawn without
    /// replacement (and multiple values emit nested subsets). With --with-replacement each
    /// requested count is drawn with replacement, so the output has exactly that many records
    /// regardless of the input size (records may repeat). Requires --r1-dst-template when
    /// multi-valued. Cannot be used with `fraction` or `record-count-per-tile`.
    #[arg(short = 'n', long, num_args = 1..)]
    pub record_count: Vec<u64>,

    /// The exact number of records to keep per tile. Reads are binned by their lane and tile
    /// extracted from the Illumina read header. Bins with fewer than this many records are
    /// discarded, and exactly this many records are randomly sampled from each retained bin.
    /// May be given more than once (requires --r1-dst-template).
    /// Cannot be used with `probability` or `record-count`.
    #[arg(long, num_args = 1..)]
    pub record_count_per_tile: Vec<u64>,

    /// Enable per-tile binning. Reads are binned by lane and tile from the Illumina read
    /// header. Can be combined with `--record-count` or `--fraction` to automatically
    /// compute the per-tile count, or use `--record-count-per-tile` to set it explicitly.
    #[arg(long)]
    pub bin_by_tile: bool,

    /// Use faster skip-ahead sampling instead of the default exact method. Skip-ahead uses
    /// exponential byte jumps and produces approximately (not exactly) the requested number
    /// of records, but avoids reading the entire file. Does not support multiple rates.
    #[arg(long)]
    pub fast: bool,

    /// Sample with replacement (bootstrap). A record may be emitted more than once. Required to
    /// oversample (a `--fraction` > 1.0). Not supported with --fast or tile binning.
    #[arg(long)]
    pub with_replacement: bool,

    /// Number of independent replicates to emit per requested rate. Each replicate uses a
    /// distinct RNG stream derived from --seed (when given), so runs are reproducible. Values
    /// > 1 require --r1-dst-template, which may contain `{sample}` to disambiguate the outputs.
    #[arg(long, default_value_t = 1)]
    pub num_samples: u32,

    /// Keep tile bins in memory instead of writing to temporary files. Uses more memory but
    /// avoids temporary disk I/O. Only used with tile binning.
    #[arg(long)]
    pub in_memory: bool,

    /// Directory for temporary tile files. Defaults to the system temp directory. Only used
    /// with tile binning when --in-memory is not set.
    #[arg(long)]
    pub temp_dir: Option<PathBuf>,

    /// Number of threads for parallel tile sampling. Only used with tile binning.
    #[arg(long, default_value_t = 1)]
    pub sampling_threads: usize,

    /// Number of threads for output compression. Only used when output is gzipped.
    #[arg(long, default_value_t = 1)]
    pub compression_threads: usize,

    /// Seed to use for the random number generator.
    #[arg(short, long)]
    pub seed: Option<u64>,

    /// Read 1 destination. Output will be gzipped if ends in `.gz`. For single-rate only;
    /// use --r1-dst-template for multi-rate output.
    #[arg(long, conflicts_with = "r1_dst_template")]
    pub r1_dst: Option<PathBuf>,

    /// Read 2 destination. Output will be gzipped if ends in `.gz`. For single-rate only;
    /// use --r2-dst-template for multi-rate output.
    #[arg(long, conflicts_with = "r2_dst_template")]
    pub r2_dst: Option<PathBuf>,

    /// Read 1 destination template. Substitutes `{quantity}` (e.g. p05, n10M, t5000) or
    /// `{value}` (raw number). Required when a multi-valued quantity flag is given.
    /// Example: `out/r1_{quantity}.fq.gz` -> `out/r1_p05.fq.gz`, `out/r1_p50.fq.gz`.
    #[arg(long, conflicts_with = "r1_dst")]
    pub r1_dst_template: Option<String>,

    /// Read 2 destination template. See --r1-dst-template.
    #[arg(long, conflicts_with = "r2_dst")]
    pub r2_dst_template: Option<String>,

    /// Read 1 source. Accepts both raw and gzipped FASTQ inputs.
    pub r1_src: PathBuf,

    /// Read 2 source. Accepts both raw and gzipped FASTQ inputs.
    pub r2_src: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_char_from_str() -> Result<(), &'static str> {
        assert_eq!("/".parse::<AsciiChar>()?, AsciiChar(b'/'));

        assert!("--".parse::<AsciiChar>().is_err());
        assert!("🪿".parse::<AsciiChar>().is_err());

        Ok(())
    }
}
