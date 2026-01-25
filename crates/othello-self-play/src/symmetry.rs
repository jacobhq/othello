use crate::async_self_play::Sample;

/// Generates all rotational and reflectional symmetries of a training sample.
///
/// This function produces the 8 dihedral symmetries of the Othello board:
/// - 4 rotations (0°, 90°, 180°, 270°),
/// - followed by a horizontal flip and 4 more rotations.
///
/// Both the board state and the policy vector are transformed consistently,
/// while the value target remains unchanged.
///
/// This is used as data augmentation to improve sample efficiency and
/// enforce symmetry invariance in training.
pub(crate) fn get_symmetries(sample: Sample) -> Vec<Sample> {
    let mut symmetries = Vec::with_capacity(8);
    let mut current_state = sample.state;
    let mut current_policy = sample.policy;

    for i in 0..8 {
        // Apply transformations
        if i == 4 {
            // After 4 rotations, flip horizontally to get reflections
            current_state = flip_state(current_state);
            current_policy = flip_policy(&current_policy);
        }

        symmetries.push(Sample {
            state: current_state,
            policy: current_policy.clone(),
            value: sample.value,
        });

        // Rotate 90 degrees for next iteration
        current_state = rotate_state(current_state);
        current_policy = rotate_policy(&current_policy);
    }
    symmetries
}

/// Rotates a board state 90 degrees clockwise.
///
/// The rotation is applied independently to both player planes of the
/// encoded board representation. The input and output use the same
/// `[player][row][col]` layout.
pub(crate) fn rotate_state(s: [[[i32; 8]; 8]; 2]) -> [[[i32; 8]; 8]; 2] {
    let mut new_s = [[[0; 8]; 8]; 2];
    for p in 0..2 {
        for r in 0..8 {
            for c in 0..8 {
                new_s[p][c][7 - r] = s[p][r][c];
            }
        }
    }
    new_s
}

/// Flips a board state horizontally (left-right).
///
/// The transformation mirrors the board along the vertical axis and is
/// applied independently to both player planes of the encoded state.
pub(crate) fn flip_state(s: [[[i32; 8]; 8]; 2]) -> [[[i32; 8]; 8]; 2] {
    let mut new_s = [[[0; 8]; 8]; 2];
    for p in 0..2 {
        for r in 0..8 {
            for c in 0..8 {
                new_s[p][r][7 - c] = s[p][r][c];
            }
        }
    }
    new_s
}

/// Rotates a flattened policy vector 90 degrees clockwise.
///
/// The input policy is assumed to be indexed in row-major order
/// (`r * 8 + c`). The returned vector corresponds to the policy distribution
/// after applying the same spatial rotation as `rotate_state`.
pub(crate) fn rotate_policy(p: &[f32]) -> Vec<f32> {
    let mut new_p = vec![0.0; 64];
    for r in 0..8 {
        for c in 0..8 {
            new_p[c * 8 + (7 - r)] = p[r * 8 + c];
        }
    }
    new_p
}

/// Flips a flattened policy vector horizontally (left-right).
///
/// The input policy is assumed to be indexed in row-major order
/// (`r * 8 + c`). The returned vector corresponds to the policy distribution
/// after applying the same horizontal reflection as `flip_state`.
pub(crate) fn flip_policy(p: &[f32]) -> Vec<f32> {
    let mut new_p = vec![0.0; 64];
    for r in 0..8 {
        for c in 0..8 {
            new_p[r * 8 + (7 - c)] = p[r * 8 + c];
        }
    }
    new_p
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a dummy sample with a single marked spot
    fn create_test_sample(row: usize, col: usize) -> Sample {
        let mut state = [[[0; 8]; 8]; 2];
        let mut policy = vec![0.0; 64];

        state[0][row][col] = 1;     // Mark a spot in the board
        policy[row * 8 + col] = 1.0; // Mark the same spot in the policy

        Sample { state, policy, value: 0.5 }
    }

    #[test]
    fn test_rotation_cycle() {
        let original = create_test_sample(0, 0); // Top-left corner
        let mut current = original.clone();

        // Rotate 4 times
        for _ in 0..4 {
            current.state = rotate_state(current.state);
            current.policy = rotate_policy(&current.policy);
        }

        // Should return to original position
        assert_eq!(current.state, original.state);
        assert_eq!(current.policy, original.policy);
    }

    #[test]
    fn test_flip_cycle() {
        let original = create_test_sample(0, 2);
        let flipped = flip_state(original.state);
        let double_flipped = flip_state(flipped);

        assert_eq!(original.state, double_flipped);

        // Check horizontal flip: (0, 2) becomes (0, 5)
        // because 7 - 2 = 5
        assert_eq!(flipped[0][0][5], 1);
    }

    #[test]
    fn test_specific_corner_rotation() {
        // Top-left (0,0) rotated 90 degrees clockwise becomes Top-right (0,7)
        let sample = create_test_sample(0, 0);
        let rotated = rotate_state(sample.state);
        let rotated_policy = rotate_policy(&sample.policy);

        assert_eq!(rotated[0][0][7], 1);
        assert_eq!(rotated_policy[7], 1.0);
    }

    #[test]
    fn test_get_symmetries_count_and_uniqueness() {
        // Use a non-symmetric point to ensure 8 unique versions
        let sample = create_test_sample(1, 2);
        let symmetries = get_symmetries(sample);

        assert_eq!(symmetries.len(), 8);

        // Check that the value (game result) is preserved across all
        for sym in &symmetries {
            assert_eq!(sym.value, 0.5);
        }

        // Optional: Ensure all 8 states are unique for an asymmetric input
        let mut seen_states = Vec::new();
        for sym in symmetries {
            assert!(!seen_states.contains(&sym.state), "Duplicate state detected in symmetries");
            seen_states.push(sym.state);
        }
    }

    #[test]
    fn test_center_invariance() {
        // The centre of rotation in an 8x8 is between indices 3 and 4.
        // However, a flip or rotation will move blocks across that line.
        // There is no single "centre" cell that stays put in an even-sized grid.
        // We test that (3,3) moves to (3,4) on a 90-degree rotation.
        let sample = create_test_sample(3, 3);
        let rotated = rotate_state(sample.state);
        assert_eq!(rotated[0][3][4], 1);
    }
}
