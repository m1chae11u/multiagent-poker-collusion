"""
Robopoker agent implementation for Texas Hold'em poker.
This module provides an agent that uses the robopoker solver to make decisions in a poker game.
"""

from typing import Tuple, Optional
from texasholdem.game.game import TexasHoldEm
from texasholdem.game.action_type import ActionType
from texasholdem.card.card import Card
import subprocess
import json
import os

class RobopokerAgent:
    """
    An agent that uses the robopoker solver to make decisions in a poker game.
    """

    def __init__(self):
        """
        Initialize the robopoker agent.
        """
        # Ensure robopoker binary exists
        if not os.path.exists("robopoker/target/release/robopoker"):
            raise RuntimeError("robopoker binary not found. Please build it first.")

    def _format_game_state(self, game: TexasHoldEm, player_id: int) -> dict:
        """
        Format the current game state for the robopoker solver.

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

        # Get position information
        dealer_pos = game.btn_loc
        current_pos = game.current_player
        num_players = len(game.players)

        # Get current betting information
        available_moves = game.get_available_moves()
        current_bet = 0
        must_call = False
        valid_actions = []

        # Determine current bet and whether player must call
        if ActionType.CALL in available_moves.action_types:
            must_call = True
            # Find the last raise or bet to determine call amount
            for action in reversed(game.hand_history[game.hand_phase].actions):
                if action.action_type == ActionType.RAISE:
                    current_bet = action.total
                    break

        # Calculate valid raise amounts
        min_raise = game.min_raise()
        max_raise = game.players[player_id].chips  # Maximum raise is player's stack

        # Get valid actions - map to robopoker's action types
        for action_type in available_moves.action_types:
            if action_type == ActionType.CHECK:
                valid_actions.append("check")
            elif action_type == ActionType.CALL:
                valid_actions.append("call")
            elif action_type == ActionType.RAISE:
                valid_actions.append("raise")
            elif action_type == ActionType.FOLD:
                valid_actions.append("fold")
            elif action_type == ActionType.ALL_IN:
                valid_actions.append("shove")  # Map ALL_IN to robopoker's Shove

        return {
            "board_cards": board_cards,
            "hole_cards": hole_cards,
            "pot_size": pot_amount,
            "stack_sizes": stack_sizes,
            "dealer_position": dealer_pos,
            "current_position": current_pos,
            "num_players": num_players,
            "current_bet": current_bet,
            "must_call": must_call,
            "valid_actions": valid_actions,
            "min_raise": min_raise,
            "max_raise": max_raise,
            "betting_round": game.hand_phase.name
        }

    def get_action(self, game: TexasHoldEm, player_id: int) -> Tuple[ActionType, Optional[int], Optional[str]]:
        """
        Get the next action from the robopoker solver.
        
        Args:
            game: The Texas Hold'em game
            player_id: The ID of the player making the decision
            
        Returns:
            A tuple of (action_type, total, reason) where total is the amount to raise to (if applicable)
            and reason is the explanation for the action (if available)
        """
        # Format the game state for the solver
        state_dict = self._format_game_state(game, player_id)
        
        # Convert state to JSON and write to temporary file
        with open("temp_state.json", "w") as f:
            json.dump(state_dict, f)
        
        try:
            # Call robopoker binary with the state file
            result = subprocess.run(
                [os.path.abspath("robopoker/target/release/robopoker"), "solve", "temp_state.json"],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"robopoker solver failed: {result.stderr}")
            
            # Parse the output
            decision = json.loads(result.stdout)
            
            # Convert the solver's decision to an action
            if decision["action"] == "check":
                return ActionType.CHECK, None, decision.get("reason", "Robopoker solver suggested check")
            elif decision["action"] == "call":
                return ActionType.CALL, None, decision.get("reason", "Robopoker solver suggested call")
            elif decision["action"] == "fold":
                return ActionType.FOLD, None, decision.get("reason", "Robopoker solver suggested fold")
            elif decision["action"] == "shove":
                return ActionType.ALL_IN, None, decision.get("reason", "Robopoker solver suggested all-in")
            elif decision["action"] == "raise":
                if "amount" in decision:
                    amount = decision["amount"]
                    # Validate raise amount
                    if amount >= game.players[player_id].chips:
                        return ActionType.ALL_IN, None, decision.get("reason", "Robopoker solver suggested all-in")
                    elif amount >= game.min_raise():
                        return ActionType.RAISE, amount, decision.get("reason", "Robopoker solver suggested raise")
                    else:
                        # If amount is invalid, use minimum raise
                        return ActionType.RAISE, game.min_raise(), decision.get("reason", "Robopoker solver suggested raise")
                else:
                    # If no amount specified, use minimum raise
                    return ActionType.RAISE, game.min_raise(), decision.get("reason", "Robopoker solver suggested raise")
            
            # Default to fold if we couldn't parse the action
            return ActionType.FOLD, None, "Could not parse robopoker solver action, defaulting to fold"
            
        finally:
            # Clean up temporary file
            if os.path.exists("temp_state.json"):
                os.remove("temp_state.json")