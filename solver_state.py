"""
Python class that matches the Rust SolverState struct for the postflop solver.
"""

class SolverState:
    """
    A class representing the state of a poker game for the postflop solver.
    This class matches the Rust struct's structure for proper interop.
    """
    def __init__(self, board_cards, hole_cards, pot_size, stack_sizes, position, betting_history):
        """
        Initialize a new SolverState.

        Args:
            board_cards (list[str]): List of board cards in string format (e.g. ['Ah', 'Kh', 'Qh'])
            hole_cards (list[str]): List of hole cards in string format (e.g. ['Jh', 'Th'])
            pot_size (int): Current pot size
            stack_sizes (list[int]): List of stack sizes for both players
            position (int): Hero's position (0 for OOP, 1 for IP)
            betting_history (list[str]): List of betting actions in string format
        """
        self.board_cards = board_cards
        self.hole_cards = hole_cards
        self.pot_size = pot_size
        self.stack_sizes = stack_sizes
        self.position = position
        self.betting_history = betting_history

    def __repr__(self):
        """Return a string representation of the SolverState."""
        return (f"SolverState(board_cards={self.board_cards}, "
                f"hole_cards={self.hole_cards}, "
                f"pot_size={self.pot_size}, "
                f"stack_sizes={self.stack_sizes}, "
                f"position={self.position}, "
                f"betting_history={self.betting_history})") 