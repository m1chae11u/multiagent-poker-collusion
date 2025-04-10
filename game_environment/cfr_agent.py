"""
CFR agent implementation for Texas Hold'em poker.
This module provides an agent that uses the postflop solver to make decisions in a poker game.
"""

from typing import Tuple, Optional
from texasholdem.game.game import TexasHoldEm
from texasholdem.game.action_type import ActionType
from texasholdem.card.card import Card
import postflop_solver
from postflop_solver import SolverState, get_optimal_action

class CFRAgent:
    """
    An agent that uses the postflop solver to make decisions in a poker game.
    """

    def __init__(self):
        """
        Initialize the CFR agent.
        """
        pass

    def _format_game_state(self, game: TexasHoldEm, player_id: int) -> dict:
        """
        Format the current game state for the solver.

        Args:
            game: The Texas Hold'em game
            player_id: The ID of the player making the decision

        Returns:
            A dictionary containing the game state for the solver
        """
        # Get player's hand
        hand = game.get_hand(player_id)
        # Format cards as "Rs" where R is rank and s is suit (no spaces)
        # Sort cards by rank in descending order (higher rank first)
        hole_cards = []
        for card in sorted(hand, key=lambda c: c.rank, reverse=True):
            rank = Card.STR_RANKS[card.rank]
            suit = Card.INT_SUIT_TO_CHAR_SUIT[card.suit]
            hole_cards.append(f"{rank}{suit}")

        # Get community cards
        community_cards = game.board
        board_cards = []
        if community_cards:
            for card in community_cards:
                rank = Card.STR_RANKS[card.rank]
                suit = Card.INT_SUIT_TO_CHAR_SUIT[card.suit]
                board_cards.append(f"{rank}{suit}")

        # Get pot information
        pot_amount = game._get_last_pot().get_total_amount()

        # Get stack sizes
        stack_sizes = [player.chips for player in game.players]

        # Get position (0 for first player, 1 for second player)
        position = player_id % 2

        # Get betting history
        betting_history = []
        current_phase = game.hand_phase
        if current_phase in game.hand_history:
            for action in game.hand_history[current_phase].actions:
                if action.action_type == ActionType.CHECK:
                    betting_history.append("check")
                elif action.action_type == ActionType.CALL:
                    betting_history.append("call")
                elif action.action_type == ActionType.RAISE:
                    betting_history.append(f"raise {action.total}")
                elif action.action_type == ActionType.FOLD:
                    betting_history.append("fold")
                elif action.action_type == ActionType.ALL_IN:
                    betting_history.append("all_in")

        return {
            "board_cards": board_cards,
            "hole_cards": hole_cards,
            "pot_size": pot_amount,
            "stack_sizes": stack_sizes,
            "position": position,
            "betting_history": betting_history
        }

    def get_action(self, game: TexasHoldEm, player_id: int) -> Tuple[ActionType, Optional[int]]:
        """
        Get the next action from the solver.
        
        Args:
            game: The Texas Hold'em game
            player_id: The ID of the player making the decision
            
        Returns:
            A tuple of (action_type, total) where total is the amount to raise to (if applicable)
        """
        # Only use solver for postflop situations
        if game.hand_phase.name == "PREFLOP":
            # For preflop, use a simple strategy (can be improved later)
            return ActionType.CALL, None

        # Format the game state for the solver
        state_dict = self._format_game_state(game, player_id)
        
        # Create a SolverState object
        state = SolverState(
            board_cards=state_dict["board_cards"],
            hole_cards=state_dict["hole_cards"],
            pot_size=state_dict["pot_size"],
            stack_sizes=state_dict["stack_sizes"],
            position=state_dict["position"],
            betting_history=state_dict["betting_history"]
        )
        
        # Get the optimal action from the solver
        decision = get_optimal_action(state)
        
        # Convert the solver's decision to an action
        if decision.action == "check":
            return ActionType.CHECK, None
        elif decision.action == "call":
            return ActionType.CALL, None
        elif decision.action == "fold":
            return ActionType.FOLD, None
        elif decision.action == "all_in":
            return ActionType.ALL_IN, None
        elif decision.action == "bet" or decision.action == "raise":
            if decision.amount is not None:
                return ActionType.RAISE, decision.amount
            else:
                # If no amount specified, use minimum raise
                return ActionType.RAISE, game.min_raise()
        
        # Default to fold if we couldn't parse the action
        return ActionType.FOLD, None 