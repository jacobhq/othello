use derive_more::{BitOr, BitOrAssign, BitAnd, Not, Shl, Shr};

/// A single 8×8 bitboard represented by a 64-bit integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, BitOr, BitOrAssign, BitAnd, Not, Shl, Shr)]
#[repr(transparent)]
pub struct BitBoard(pub u64);

impl BitBoard {
    /// Converts a (row, col) coordinate pair into a single bit index (0–63).
    #[inline]
    fn index(row: usize, col: usize) -> usize {
        row * 8 + col
    }

    /// Returns a bit mask (`u64`) with a single bit set at the given (row, col).
    #[inline]
    fn mask(row: usize, col: usize) -> u64 {
        1u64 << Self::index(row, col)
    }

    /// Checks whether the bit at (row, col) is set.
    pub fn get(&self, row: usize, col: usize) -> bool {
        self.0 & Self::mask(row, col) != 0
    }

    /// Sets the bit at (row, col).
    pub fn set(&mut self, row: usize, col: usize) {
        self.0 |= Self::mask(row, col);
    }

    /// Clears the bit at (row, col).
    pub fn clear(&mut self, row: usize, col: usize) {
        self.0 &= !Self::mask(row, col);
    }
}

impl From<BitBoard> for u64 {
    fn from(bb: BitBoard) -> u64 { bb.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_and_mask() {
        assert_eq!(BitBoard::index(0, 0), 0);
        assert_eq!(BitBoard::index(7, 7), 63);
        assert_eq!(BitBoard::mask(0, 0), 1);
        assert_eq!(BitBoard::mask(0, 1), 1 << 1);
        assert_eq!(BitBoard::mask(7, 7), 1u64 << 63);
    }

    #[test]
    fn test_set_and_get() {
        let mut bb = BitBoard(0);
        bb.set(3, 4);
        assert!(bb.get(3, 4));
        assert!(!bb.get(0, 0));
    }

    #[test]
    fn test_clear() {
        let mut bb = BitBoard(0);
        bb.set(2, 2);
        assert!(bb.get(2, 2));
        bb.clear(2, 2);
        assert!(!bb.get(2, 2));
    }

    #[test]
    fn test_multiple_bits() {
        let mut bb = BitBoard(0);
        bb.set(0, 0);
        bb.set(7, 7);
        assert!(bb.get(0, 0));
        assert!(bb.get(7, 7));
        assert_eq!(bb.0.count_ones(), 2);
    }
}
