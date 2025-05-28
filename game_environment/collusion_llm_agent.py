"""
Collusion LLM agent implementation for Texas Hold'em poker.
This module provides an agent that uses a Language Model to make decisions in a poker game.

Example of what the final prompt looks like:
```
You are playing Texas Hold'em poker. Here is the current game state:

Current game state:
- Your position: 2
- Dealer position: 0
- Small blind position: 1
- Big blind position: 2
- Your hand: A♠, K♥
- Community cards: Q♣, J♦, 10♠
- Current phase: FLOP
- Pot amount: 100
- Your chips: 200
- Chips to call: 10
- Minimum raise: 20

Player positions and chips:
Position 0: 150 chips (Active)
Position 1: 180 chips (Active)
Position 2: 200 chips (Active)
Position 3: 120 chips (Folded)

Betting history for current round:
Position 0: CHECK
Position 1: CHECK

Available actions:
CHECK: Check (pass the action without betting)
CALL: Call (match the current bet of 10 chips)
RAISE: Raise (increase the bet, minimum raise is 20 chips, maximum is 200 chips)
Bet choices:
- 33% of pot (33 chips)
- 50% of pot (50 chips)
- 75% of pot (75 chips)
- 100% of pot (100 chips)
- 150% of pot (150 chips)
- All-in (200 chips)
FOLD: Fold (give up the hand and exit the pot)

What action would you take? Respond with just the action name and amount (if raising) in the format:
ACTION [amount]

For example:
CALL
RAISE 100
FOLD
CHECK
ALL_IN

Your response:
```
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


class CollusionLLMAgent:
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

        # Get player positions and chips
        positions_info = []
        for pos in range(len(game.players)):
            player = game.players[pos]
            state = "Folded" if player.state == PlayerState.OUT else "Active"
            positions_info.append(f"Position {pos}: {player.chips} chips ({state})")

        # Get betting history
        betting_history = []
        if hasattr(game, 'history') and game.history:
            for action in game.history.get_current_round():
                if isinstance(action, PlayerAction):
                    betting_history.append(
                        f"Position {action.player_id}: {action.action_type.name} {action.total if action.total else ''}"
                    )

        # Get dealer and blind positions
        dealer_pos = (game.button % len(game.players)) if hasattr(game, 'button') else None
        sb_pos = (dealer_pos + 1) % len(game.players) if dealer_pos is not None else None
        bb_pos = (dealer_pos + 2) % len(game.players) if dealer_pos is not None else None

        # Format the state
        state = f"""
Current game state:
- Your position: {player_id}
- Dealer position: {dealer_pos}
- Small blind position: {sb_pos}
- Big blind position: {bb_pos}
- Your hand: {hand_str}
- Community cards: {community_str}
- Current phase: {phase}
- Pot amount: {pot_amount}
- Your chips: {player_chips}
- Chips to call: {chips_to_call}
- Minimum raise: {min_raise}

Player positions and chips:
{chr(10).join(positions_info)}

Betting history for current round:
{chr(10).join(betting_history) if betting_history else "No betting history yet"}
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

        # Calculate pot-based betting suggestions
        pot_amount = game._get_last_pot().get_total_amount()
        min_raise = game.min_raise()
        player_chips = game.players[player_id].chips

        # Add all available actions from the MoveIterator
        for action_type in moves.action_types:
            if action_type == ActionType.CHECK:
                actions[ActionType.CHECK] = "Check (pass the action without betting)"
            elif action_type == ActionType.CALL:
                actions[ActionType.CALL] = (
                    f"Call (match the current bet of {game.chips_to_call(player_id)} chips)"
                )
            elif action_type == ActionType.RAISE:
                # Calculate common bet sizes as percentages of pot
                pot_percentages = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0]
                bet_sizes = []
                for percentage in pot_percentages:
                    suggested_amount = int(pot_amount * percentage)
                    if min_raise <= suggested_amount <= player_chips:
                        bet_sizes.append(f"{int(percentage * 100)}% of pot ({suggested_amount} chips)")
                
                # Add all-in as a suggestion if it's a valid option
                if player_chips >= min_raise:
                    bet_sizes.append(f"All-in ({player_chips} chips)")
                
                actions[ActionType.RAISE] = (
                    f"Raise (increase the bet, minimum raise is {min_raise} chips, maximum is {player_chips} chips)\n"
                    f"Bet choices:\n" + "\n".join(f"- {size}" for size in bet_sizes)
                )
            elif action_type == ActionType.FOLD:
                actions[ActionType.FOLD] = "Fold (give up the hand and exit the pot)"
            elif action_type == ActionType.ALL_IN:
                actions[ActionType.ALL_IN] = (
                    f"All-in (bet all {player_chips} chips)"
                )

        return actions

    def get_action(self, game: TexasHoldEm, player_id: int) -> Tuple[ActionType, Optional[int], Optional[str]]:
        """
        Get the next action from the LLM.
        
        Args:
            game: The Texas Hold'em game
            player_id: The ID of the player making the decision
            
        Returns:
            A tuple of (action_type, total, reason) where:
                - action_type is the type of action to take
                - total is the amount to raise to (if applicable)
                - reason is the explanation for the action (None for LLM agents)
        """
        # Format the game state for the LLM
        state_str = self._format_game_state(game, player_id)
        
        # Get available actions
        available_actions = self._get_available_actions(game, player_id)
        
        # Create the prompt for the LLM
        prompt = f"""You are playing Texas Hold'em poker. Here is the current game state:

{state_str}

Available actions:
{', '.join(f'{action.name}: {desc}' for action, desc in available_actions.items())}

What action would you take? Respond with just the action name and amount (if raising) in the format:
ACTION [amount]

For example:
CALL
RAISE 100
FOLD
CHECK
ALL_IN

Your response:"""
        
        # Get the response from the LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a poker player making decisions in a Texas Hold'em game. Respond with just the action and amount (if raising)."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=10
        )
        
        # Parse the response
        action_str = response.choices[0].message.content.strip().upper()
        
        # Extract action type and amount
        match = re.match(r"(\w+)(?:\s+(\d+))?", action_str)
        if not match:
            return ActionType.FOLD, None, None
            
        action_type_str, amount_str = match.groups()
        
        # Convert action type string to ActionType enum
        try:
            action_type = ActionType[action_type_str]
        except KeyError:
            return ActionType.FOLD, None, None
            
        # Convert amount string to int if present
        total = int(amount_str) if amount_str else None
        
        # Validate the action
        if action_type not in available_actions:
            return ActionType.FOLD, None, None
            
        # Return the action without a reason
        return action_type, total, None 