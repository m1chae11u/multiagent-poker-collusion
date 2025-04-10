from typing import Dict, List, Optional, Tuple
import json
from game_environment.poker_game import PokerGame
from game_environment.player import Player

class SolverAgent(Player):
    """A poker agent that uses the poker solver engine for decision making."""
    
    def __init__(self, name: str, solver_path: str):
        """Initialize the solver agent.
        
        Args:
            name: The name of the agent
            solver_path: Path to the compiled poker solver library
        """
        super().__init__(name)
        # TODO: Load the Rust solver library using PyO3
        self.solver = None  # Will be initialized with the Rust solver
        
    def get_action(self, game_state: Dict) -> Tuple[str, Optional[int], str]:
        """Get the next action using the poker solver.
        
        Args:
            game_state: The current game state
            
        Returns:
            Tuple of (action_name, amount, reason)
        """
        # Convert game state to solver format
        solver_state = self._convert_game_state(game_state)
        
        # Get solver's decision
        action, amount, reason = self._get_solver_decision(solver_state)
        
        # Convert solver action to game action format
        return self._convert_solver_action(action, amount, reason)
    
    def _convert_game_state(self, game_state: Dict) -> Dict:
        """Convert the game state to the solver's format."""
        # TODO: Implement conversion from game state to solver state
        # This will need to map:
        # - Board cards
        # - Hole cards
        # - Pot size
        # - Stack sizes
        # - Position
        # - Betting history
        pass
    
    def _get_solver_decision(self, solver_state: Dict) -> Tuple[str, Optional[int], str]:
        """Get the decision from the solver.
        
        Args:
            solver_state: The game state in solver format
            
        Returns:
            Tuple of (action, amount, reason)
        """
        # TODO: Call the Rust solver to get the optimal action
        # This will need to:
        # 1. Initialize the solver with the current state
        # 2. Run the solver to get the optimal action
        # 3. Extract the action and reasoning
        pass
    
    def _convert_solver_action(self, action: str, amount: Optional[int], reason: str) -> Tuple[str, Optional[int], str]:
        """Convert the solver's action to the game's action format."""
        # Map solver actions to game actions
        action_map = {
            'Fold': 'fold',
            'Check': 'check',
            'Call': 'call',
            'Bet': 'raise',
            'Raise': 'raise',
            'AllIn': 'all_in'
        }
        
        return (action_map[action], amount, reason) 