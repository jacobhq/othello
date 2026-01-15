//! Binary serialization for self-play training samples.
//!
//! This module defines a writer for serializing [`Sample`] values into a
//! compact, versioned binary format suitable for long-term storage and
//! high-throughput training pipelines (e.g., neural network training).
//!
//! ## File Format Overview
//!
//! All numeric values are encoded in **little-endian** byte order.
//! The file layout is strictly sequential and contains no padding.
//!
//! ```text
//! ┌──────────────────────────────────────────────┐
//! │ Magic        : u32 ("OTHL")                  │
//! │ Version      : u32                           │
//! │ Sample count : u32                           │
//! ├──────────────────────────────────────────────┤
//! │ Sample 0                                     │
//! │   State      : 2 × 8 × 8 i32                 │
//! │   Policy     : 64 f32                        │
//! │   Value      : f32                           │
//! ├──────────────────────────────────────────────┤
//! │ Sample 1                                     │
//! │   ...                                        │
//! └──────────────────────────────────────────────┘
//! ```
//!
//! ### Fields
//!
//! * **Magic**
//!   * ASCII `"OTHL"` (`0x4F54484C`) used to identify the file type.
//! * **Version**
//!   * Format version number (currently `1`).
//!   * Allows backward-compatible format evolution.
//! * **Sample count**
//!   * Number of samples stored in the file.
//!   * Enables integrity checking and pre-allocation when reading.
//! * **State**
//!   * Two 8×8 integer planes representing the board state.
//! * **Policy**
//!   * A probability distribution over 64 possible actions.
//! * **Value**
//!   * Scalar evaluation target (e.g., game outcome).
//!
//! ## Design Goals
//!
//! * **Deterministic layout** — byte-for-byte reproducibility.
//! * **Streaming-friendly** — samples are written sequentially.
//! * **Language-agnostic** — trivial to parse from C/C++/Python.
//! * **Forward-compatible** — versioned header enables evolution.
//!
//! ## Error Handling
//!
//! The current implementation panics on I/O errors or malformed samples.
//! This is intentional for early development and testing. Production
//! callers may wish to wrap the writer in a `Result`-returning API.
//!
//! ## Testing
//!
//! The accompanying tests validate the exact binary layout by reading
//! back raw bytes and decoding each field manually, ensuring that any
//! future changes to the format are caught immediately.
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use crate::self_play::Sample;

/// Writes a collection of self-play samples to a binary file in a fixed,
/// versioned format.
///
/// # Binary Format
///
/// All values are encoded in **little-endian** byte order.
///
/// ```text
/// ┌──────────────────────────────────────────────┐
/// │ Magic        : u32 ("OTHL")                  │
/// │ Version      : u32 (currently 1)             │
/// │ Sample count : u32                           │
/// ├──────────────────────────────────────────────┤
/// │ Sample 0                                     │
/// │   State      : 2 × 8 × 8 i32                 │
/// │   Policy     : 64 f32                        │
/// │   Value      : f32                           │
/// ├──────────────────────────────────────────────┤
/// │ Sample 1                                     │
/// │   ...                                        │
/// └──────────────────────────────────────────────┘
/// ```
///
/// # Arguments
///
/// * `path` - Path to the file that will be created or truncated.
/// * `samples` - Slice of [`Sample`] values to serialize.
///
/// # Panics
///
/// * Panics if the file cannot be created or written.
/// * Panics if any sample's policy vector does not have exactly 64 entries.
///
/// # Notes
///
/// * The sample count field is technically optional but strongly recommended,
///   as it allows readers to validate file integrity.
/// * This function performs no buffering beyond the OS defaults; callers
///   writing large datasets may want to wrap the file in a `BufWriter`.
pub fn write_samples(path: &PathBuf, samples: &[Sample]) {
    let mut f = File::create(path).unwrap();
    let mut f = BufWriter::new(f);

    // Magic: "OTHL"
    f.write_all(&0x4F54484Cu32.to_le_bytes()).unwrap();

    // Version
    f.write_all(&1u32.to_le_bytes()).unwrap();

    // Number of samples (optional but strongly recommended)
    f.write_all(&(samples.len() as u32).to_le_bytes()).unwrap();

    for s in samples {

        // Board state: 2 × 8 × 8 i32
        for plane in 0..2 {
            for row in 0..8 {
                for col in 0..8 {
                    f.write_all(&s.state[plane][row][col].to_le_bytes())
                        .unwrap();
                }
            }
        }

        // Policy: 64 f32
        assert_eq!(s.policy.len(), 64);
        for &p in &s.policy {
            f.write_all(&p.to_le_bytes()).unwrap();
        }

        // Value: f32
        f.write_all(&s.value.to_le_bytes()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::fs;

    /// Constructs a deterministic [`Sample`] for testing purposes.
    ///
    /// The board state is filled with predictable integer values derived
    /// from `(plane, row, col)` so that serialization order and correctness
    /// can be validated byte-for-byte.
    ///
    /// # Arguments
    ///
    /// * `value` - The scalar value to store in the sample.
    ///
    /// # Returns
    ///
    /// A fully-initialised [`Sample`] with:
    /// * A 2×8×8 board state
    /// * A policy vector of length 64
    /// * The provided value
    fn make_sample(value: f32) -> Sample {
        let mut state = [[[0i32; 8]; 8]; 2];

        // Fill with deterministic data
        for plane in 0..2 {
            for row in 0..8 {
                for col in 0..8 {
                    state[plane][row][col] =
                        (plane * 100 + row * 10 + col) as i32;
                }
            }
        }

        let policy = (0..64).map(|i| i as f32 / 64.0).collect();

        Sample {
            state,
            policy,
            value,
        }
    }

    /// Verifies that [`write_samples`] produces the expected binary layout.
    ///
    /// This test performs a full round-trip validation by:
    /// * Writing multiple samples to a temporary file
    /// * Reading the raw bytes back
    /// * Manually decoding each field
    /// * Asserting exact equality with the original data
    ///
    /// The test also checks:
    /// * Magic number correctness
    /// * Version correctness
    /// * Sample count correctness
    /// * Absence of trailing or missing bytes
    #[test]
    fn test_write_samples_binary_format() {
        let sample1 = make_sample(0.5);
        let sample2 = make_sample(-1.0);
        let samples = vec![sample1.clone(), sample2.clone()];

        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();

        write_samples(&path, &samples);

        let bytes = fs::read(path).unwrap();
        let mut offset = 0;

        // Magic verification
        let magic = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        assert_eq!(magic, 0x4F54484C);

        // Version of binary format
        let version = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        assert_eq!(version, 1);

        // Sample count
        let count = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        assert_eq!(count, 2);

        for sample in samples {
            // State
            for plane in 0..2 {
                for row in 0..8 {
                    for col in 0..8 {
                        let v = i32::from_le_bytes(
                            bytes[offset..offset + 4].try_into().unwrap()
                        );
                        offset += 4;
                        assert_eq!(v, sample.state[plane][row][col]);
                    }
                }
            }

            // Policy
            for expected in sample.policy {
                let v = f32::from_le_bytes(
                    bytes[offset..offset + 4].try_into().unwrap()
                );
                offset += 4;
                assert_eq!(v, expected);
            }

            // Value
            let value = f32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap()
            );
            offset += 4;
            assert_eq!(value, sample.value);
        }

        // Ensure no trailing garbage
        assert_eq!(offset, bytes.len());
    }
}

