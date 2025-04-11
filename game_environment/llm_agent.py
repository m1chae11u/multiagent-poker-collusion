"""
LLM agent implementation for Texas Hold'em poker.
This module provides an agent that uses a Language Model to make decisions in a poker game.
"""

import os
import re
from typing import Tuple, Optional, Dict, Any
import openai
from dotenv import load_dotenv
from texasholdem.game.game import TexasHoldEm
from texasholdem.game.action_type import ActionType
from texasholdem.game.player_state import PlayerState
from texasholdem.card.card import Card


class LLMAgent:
    """
    An agent that uses a Language Model to make decisions in a poker game.
    """

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize the LLM agent.

        Args:
            model: The model name to use
            api_key: The API key. If None, will try to get from .env file
        """
        # Load environment variables from .env file
        load_dotenv()
        
        self.model = model
        if api_key:
            openai.api_key = api_key
        elif "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError("API key not provided and not found in .env file")
            
        # Initialize the client
        self.client = openai.OpenAI()

    def _format_game_state(self, game: TexasHoldEm, player_id: int) -> str:
        """
        Format the current game state as a string for the LLM.

        Args:
            game: The Texas Hold'em game
            player_id: The ID of the player making the decision

        Returns:
            A string representation of the game state
        """
        # Get player's hand
        hand = game.get_hand(player_id)
        hand_str = ", ".join([card.pretty_string for card in hand])

        # Get community cards
        community_cards = game.board
        community_str = (
            ", ".join([card.pretty_string for card in community_cards])
            if community_cards
            else "None"
        )

        # Get pot information
        pot_amount = game._get_last_pot().get_total_amount()

        # Get player's chips
        player_chips = game.players[player_id].chips

        # Get current phase
        phase = game.hand_phase.name

        # Get betting information
        chips_to_call = game.chips_to_call(player_id)
        min_raise = game.min_raise()

        # Format the state
        state = f"""
Current game state:
- Your hand: {hand_str}
- Community cards: {community_str}
- Current phase: {phase}
- Pot amount: {pot_amount}
- Your chips: {player_chips}
- Chips to call: {chips_to_call}
- Minimum raise: {min_raise}
"""
        return state

    def _get_available_actions(
        self, game: TexasHoldEm, player_id: int
    ) -> Dict[ActionType, str]:
        """
        Get the available actions for the player.

        Args:
            game: The Texas Hold'em game
            player_id: The ID of the player making the decision

        Returns:
            A dictionary mapping action types to descriptions
        """
        moves = game.get_available_moves()
        actions = {}

        # Add all available actions from the MoveIterator
        for action_type in moves.action_types:
            if action_type == ActionType.CHECK:
                actions[ActionType.CHECK] = "Check (pass the action)"
            elif action_type == ActionType.CALL:
                actions[ActionType.CALL] = (
                    f"Call (match the current bet of {game.chips_to_call(player_id)} chips)"
                )
            elif action_type == ActionType.RAISE:
                min_raise = game.min_raise()
                actions[ActionType.RAISE] = (
                    f"Raise (increase the bet, minimum raise is {min_raise} chips)"
                )
            elif action_type == ActionType.FOLD:
                actions[ActionType.FOLD] = "Fold (give up the hand)"

        # Add ALL_IN if the player has chips
        if game.players[player_id].chips > 0:
            actions[ActionType.ALL_IN] = (
                f"All_in (bet all {game.players[player_id].chips} chips)"
            )

        return actions

    def get_action(self, game: TexasHoldEm, player_id: int) -> Tuple[ActionType, Optional[int]]:
        """
        Get the next action from the LLM.
        
        Args:
            game: The Texas Hold'em game
            player_id: The ID of the player making the decision
            
        Returns:
            A tuple of (action_type, total) where total is the amount to raise to (if applicable)
        """
        # Format the game state for the prompt
        formatted_state = self._format_game_state(game, player_id)
        
        # Get available actions
        available_actions = self._get_available_actions(game, player_id)
        actions_str = "\n".join([f"- {action.value}: {desc}" for action, desc in available_actions.items()])
        
        # Create the prompt
        prompt = f"""You are playing Texas Hold'em poker. Here is the current game state:

{formatted_state}

Available actions:
{actions_str}

Choose one of the available actions. For raise, specify the amount."""

        # Get the response from the model
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        
        # Get the response text
        response_text = response.choices[0].message.content.lower()
        
        # Parse the response
        if "fold" in response_text:
            return ActionType.FOLD, None
        elif "check" in response_text:
            return ActionType.CHECK, None
        elif "call" in response_text:
            return ActionType.CALL, None
        elif "all_in" in response_text:
            return ActionType.ALL_IN, None
        elif "raise" in response_text:
            # Try to extract the raise amount
            raise_match = re.search(r'raise\s+(\d+)', response_text)
            if raise_match:
                try:
                    raise_amount = int(raise_match.group(1))
                    return ActionType.RAISE, raise_amount
                except ValueError:
                    pass
            
            # If we couldn't extract a specific amount, use the minimum raise
            min_raise = game.min_raise()
            return ActionType.RAISE, min_raise
        
        # Default to fold if we couldn't parse the response
        return ActionType.FOLD, None
