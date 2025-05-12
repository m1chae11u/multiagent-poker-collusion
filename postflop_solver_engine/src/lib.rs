//! An open-source postflop solver library.
//!
//! # Examples
//!
//! See the [examples] directory.
//!
//! [examples]: https://github.com/b-inary/postflop-solver/tree/main/examples
//!
//! # Implementation details
//! - **Algorithm**: The solver uses the state-of-the-art [Discounted CFR] algorithm.
//!   Currently, the value of γ is set to 3.0 instead of the 2.0 recommended in the original paper.
//!   Also, the solver resets the cumulative strategy when the number of iterations is a power of 4.
//! - **Performance**: The solver engine is highly optimized for performance with maintainable code.
//!   The engine supports multithreading by default, and it takes full advantage of unsafe Rust in hot spots.
//!   The developer reviews the assembly output from the compiler and ensures that SIMD instructions are used as much as possible.
//!   Combined with the algorithm described above, the performance surpasses paid solvers such as PioSOLVER and GTO+.
//! - **Isomorphism**: The solver does not perform any abstraction.
//!   However, isomorphic chances (turn and river deals) are combined into one.
//!   For example, if the flop is monotone, the three non-dealt suits are isomorphic,
//!   allowing us to skip the calculation for two of the three suits.
//! - **Precision**: 32-bit floating-point numbers are used in most places.
//!   When calculating summations, temporary values use 64-bit floating-point numbers.
//!   There is also a compression option where each game node stores the values
//!   by 16-bit integers with a single 32-bit floating-point scaling factor.
//! - **Bunching effect**: At the time of writing, this is the only implementation that can handle the bunching effect.
//!   It supports up to four folded players (6-max game).
//!   The implementation correctly counts the number of card combinations and does not rely on heuristics
//!   such as manipulating the probability distribution of the deck.
//!   Note, however, that enabling the bunching effect increases the time complexity
//!   of the evaluation at the terminal nodes and slows down the computation significantly.
//!
//! [Discounted CFR]: https://arxiv.org/abs/1809.04040
//!
//! # Crate features
//! - `bincode`: Uses [bincode] crate (2.0.0-rc.3) to serialize and deserialize the `PostFlopGame` struct.
//!   This feature is required to save and load the game tree.
//!   Enabled by default.
//! - `custom-alloc`: Uses custom memory allocator in solving process (only available in nightly Rust).
//!   It significantly reduces the number of calls of the default allocator,
//!   so it is recommended to use this feature when the default allocator is not so efficient.
//!   Note that this feature assumes that, at most, only one instance of `PostFlopGame` is available
//!   when solving in a program.
//!   Disabled by default.
//! - `rayon`: Uses [rayon] crate for parallelization.
//!   Enabled by default.
//! - `zstd`: Uses [zstd] crate to compress and decompress the game tree.
//!   This feature is required to save and load the game tree with compression.
//!   Disabled by default.
//!
//! [bincode]: https://github.com/bincode-org/bincode
//! [rayon]: https://github.com/rayon-rs/rayon
//! [zstd]: https://github.com/gyscos/zstd-rs

#![cfg_attr(feature = "custom-alloc", feature(allocator_api))]

#[cfg(feature = "custom-alloc")]
mod alloc;

#[cfg(feature = "bincode")]
mod file;

mod action_tree;
mod atomic_float;
mod bet_size;
mod bunching;
mod card;
mod game;
mod hand;
mod hand_table;
mod interface;
mod mutex_like;
mod range;
mod sliceop;
mod solver;
mod utility;

#[cfg(feature = "bincode")]
pub use file::*;

pub use action_tree::*;
pub use bet_size::*;
pub use bunching::*;
pub use card::*;
pub use game::*;
pub use interface::*;
pub use mutex_like::*;
pub use range::*;
pub use solver::*;
pub use utility::*;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

#[cfg(feature = "python")]
#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SolverState {
    #[pyo3(get, set)]
    pub board_cards: Vec<String>,
    #[pyo3(get, set)]
    pub hole_cards: Vec<String>,
    #[pyo3(get, set)]
    pub pot_size: i32,
    #[pyo3(get, set)]
    pub stack_sizes: Vec<i32>,
    #[pyo3(get, set)]
    pub position: i32,
    #[pyo3(get, set)]
    pub current_bet: i32,
    #[pyo3(get, set)]
    pub must_call: bool,
    #[pyo3(get, set)]
    pub valid_actions: Vec<String>,
    #[pyo3(get, set)]
    pub betting_round: String,
}

#[cfg(feature = "python")]
#[pymethods]
impl SolverState {
    #[new]
    fn new(
        board_cards: Vec<String>,
        hole_cards: Vec<String>,
        pot_size: i32,
        stack_sizes: Vec<i32>,
        position: i32,
        current_bet: i32,
        must_call: bool,
        valid_actions: Vec<String>,
        betting_round: String,
    ) -> Self {
        SolverState {
            board_cards,
            hole_cards,
            pot_size,
            stack_sizes,
            position,
            current_bet,
            must_call,
            valid_actions,
            betting_round,
        }
    }
}

#[cfg(feature = "python")]
#[pyclass]
#[derive(Debug, Serialize, Deserialize)]
pub struct SolverDecision {
    #[pyo3(get, set)]
    pub action: String,
    #[pyo3(get, set)]
    pub amount: Option<i32>,
    #[pyo3(get, set)]
    pub reason: String,
}

#[cfg(feature = "python")]
#[pyfunction]
fn get_optimal_action(state: &PyAny) -> PyResult<SolverDecision> {
    // Convert Python state to Rust state
    let solver_state: SolverState = state.extract()?;
    
    // Parse board cards
    let flop = if solver_state.board_cards.len() >= 3 {
        let flop_str = format!("{}{}{}", 
            solver_state.board_cards[0], 
            solver_state.board_cards[1], 
            solver_state.board_cards[2]
        );
        flop_from_str(&flop_str).unwrap_or([NOT_DEALT, NOT_DEALT, NOT_DEALT])
    } else {
        [NOT_DEALT, NOT_DEALT, NOT_DEALT]
    };
    
    let turn = if solver_state.board_cards.len() >= 4 {
        card_from_str(&solver_state.board_cards[3]).unwrap_or(NOT_DEALT)
    } else {
        NOT_DEALT
    };
    
    let river = if solver_state.board_cards.len() >= 5 {
        card_from_str(&solver_state.board_cards[4]).unwrap_or(NOT_DEALT)
    } else {
        NOT_DEALT
    };
    
    // Parse hole cards
    let hole_cards = format!("{}{}", 
        solver_state.hole_cards[0], 
        solver_state.hole_cards[1]
    );
    
    // Create card config
    let card_config = CardConfig {
        range: [hole_cards.parse().unwrap(), "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+".parse().unwrap()],
        flop,
        turn,
        river,
    };
    
    // Create tree config
    let bet_sizes = BetSizeOptions::try_from(("60%, e, a", "2.5x")).unwrap();
    let tree_config = TreeConfig {
        initial_state: if river != NOT_DEALT { BoardState::River } 
                       else if turn != NOT_DEALT { BoardState::Turn }
                       else { BoardState::Flop },
        starting_pot: solver_state.pot_size,
        effective_stack: solver_state.stack_sizes[0].min(solver_state.stack_sizes[1]),
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: Some(DonkSizeOptions::try_from("50%").unwrap()),
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
    };
    
    // Build and solve the game
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.allocate_memory(false);
    
    // Solve with just 1 iteration for faster speed
    let max_num_iterations = 50;
    let target_exploitability = game.tree_config().starting_pot as f32 * 0.01;
    solve(&mut game, max_num_iterations, target_exploitability, true);
    
    // Get available actions and their probabilities from the solver
    let actions = game.available_actions();
    let strategy = if game.is_compression_enabled() {
        regret_matching_compressed(game.node().regrets_compressed(), actions.len())
    } else {
        regret_matching(game.node().regrets(), actions.len())
    };
    
    // Filter actions based on valid_actions from the solver state
    let valid_actions_set: std::collections::HashSet<String> = solver_state.valid_actions.iter().cloned().collect();
    
    // Find the action with highest probability that is in the valid_actions list
    let (action, amount, reason) = actions.iter()
        .zip(strategy.iter())
        .filter_map(|(action, &prob)| {
            let (action_str, amount) = match action {
                Action::Check => ("check", None),
                Action::Bet(amt) => ("bet", Some(*amt)),
                Action::Raise(amt) => ("raise", Some(*amt)),
                Action::AllIn(_) => ("all_in", Some(solver_state.stack_sizes[solver_state.position as usize])),
                Action::None => ("none", None),
                Action::Fold => ("fold", None),
                Action::Call => ("call", None),
                Action::Chance(_) => ("chance", None),
            };
            
            if valid_actions_set.contains(&action_str.to_string()) {
                Some((action_str.to_string(), amount, prob))
            } else {
                None
            }
        })
        .max_by(|(_, _, prob_a): &(String, Option<i32>, f32), (_, _, prob_b): &(String, Option<i32>, f32)| 
            prob_a.partial_cmp(prob_b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(action, amount, prob)| {
            (action, amount, format!("Optimal action based on GTO strategy (probability: {:.2}%)", prob * 100.0))
        })
        .unwrap_or(("fold".to_string(), None, format!("Defaulting to fold because no valid actions were found in the solver's available actions. Valid actions: {:?}", solver_state.valid_actions)));
    
    // Return the decision
    Ok(SolverDecision {
        action,
        amount,
        reason,
    })
}

#[cfg(feature = "python")]
#[pymodule]
fn postflop_solver(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_optimal_action, m)?)?;
    m.add_class::<SolverState>()?;
    Ok(())
}
