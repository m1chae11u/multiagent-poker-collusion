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
from texasholdem.game.hand_phase import HandPhase
import json


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
        hand_str = ", ".join([f"{Card.STR_RANKS[card.rank]}{Card.INT_SUIT_TO_CHAR_SUIT[card.suit]}" for card in hand])

        # Get community cards
        community_cards = game.board
        community_str = (
            ", ".join([f"{Card.STR_RANKS[card.rank]}{Card.INT_SUIT_TO_CHAR_SUIT[card.suit]}" for card in community_cards])
            if community_cards
            else "None"
        )

        # Get pot information
        pot_amount = game._get_last_pot().get_total_amount()

        # Get current phase
        phase = game.hand_phase.name

        # Get betting information
        chips_to_call = game.chips_to_call(player_id)
        min_raise = game.min_raise()

        # Get player positions and chips
        positions_info = []
        num_players = len(game.players)
        
        # Define position names based on number of players
        position_names = {
            2: ["SB", "BB"],
            3: ["SB", "BB", "UTG"],
            4: ["SB", "BB", "UTG", "CO"],
            5: ["SB", "BB", "UTG", "MP", "CO"],
            6: ["SB", "BB", "UTG", "MP", "CO", "BTN"],            
        }
        
        # Get position names for current number of players
        current_positions = position_names.get(num_players, [f"P{i}" for i in range(num_players)])
        
        # Rotate positions based on button location
        btn_loc = game.btn_loc
        rotated_positions = current_positions[btn_loc:] + current_positions[:btn_loc]
        
        for pos in range(len(game.players)):
            player = game.players[pos]
            state = "Folded" if player.state == PlayerState.OUT else "Active"
            position_name = rotated_positions[pos]
            positions_info.append(f"Position {pos} ({position_name}): {player.chips} chips ({state})")

        # Get complete betting history
        betting_history = []
        if game.hand_history:
            for hand_phase in [HandPhase.PREFLOP, HandPhase.FLOP, HandPhase.TURN, HandPhase.RIVER]:
                if hand_phase in game.hand_history and game.hand_history[hand_phase]:
                    betting_history.append(f"\n{hand_phase.name}:")
                    for action in game.hand_history[hand_phase].actions:
                        position_name = rotated_positions[action.player_id]
                        betting_history.append(
                            f"Position {action.player_id} ({position_name}): {action.action_type.name} {action.total if action.total else ''}"
                        )

        # Format the state
        state = f"""
Current game state:
- Your position: {player_id} ({rotated_positions[player_id]})
- Small blind position: {game.sb_loc} ({rotated_positions[game.sb_loc]})
- Big blind position: {game.bb_loc} ({rotated_positions[game.bb_loc]})
- Your hand: {hand_str}
- Community cards: {community_str}
- Current phase: {phase}
- Pot amount: {pot_amount}
- Your chips: {game.players[player_id].chips}
- Chips to call: {chips_to_call}
- Minimum raise: {min_raise}

Player positions and chips:
{chr(10).join(positions_info)}

Betting history:
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
        previous_bet = game.player_bet_amount(player_id)

        # Add all available actions from the MoveIterator
        for action_type in moves.action_types:
            if action_type == ActionType.CHECK:
                actions[ActionType.CHECK] = "Check (pass the action without betting)"
            elif action_type == ActionType.CALL:
                actions[ActionType.CALL] = (
                    f"Call (match the current bet of {game.chips_to_call(player_id)} chips)"
                )
            elif action_type == ActionType.RAISE:
                # Calculate bet sizes based on pot and previous bet
                bet_sizes = []
                
                # Pot-based bet sizes
                pot_percentages = [0.33, 0.5, 0.66, 1.25]
                for percentage in pot_percentages:
                    suggested_amount = int(pot_amount * percentage)
                    if min_raise <= suggested_amount <= player_chips:
                        bet_sizes.append(f"{int(percentage * 100)}% of pot ({suggested_amount} chips)")
                
                # Previous bet multiplier
                if previous_bet > 0:
                    suggested_amount = int(previous_bet * 2.5)
                    if min_raise <= suggested_amount <= player_chips:
                        bet_sizes.append(f"2.5x previous bet ({suggested_amount} chips)")
                
                # Add all-in if it would be less than 20% of the pot
                if player_chips >= min_raise:
                    remaining_chips = player_chips - min_raise
                    if remaining_chips < pot_amount * 0.2:
                        bet_sizes.append(f"All-in ({player_chips} chips)")
                    elif len(bet_sizes) == 0:  # If no other valid bets, add all-in
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
        prompt = f"""You are a Texas Hold'em poker player in a game that has a 52-card deck, and 3 betting rounds (preflop/flop/river).
Your goal is to maximise your own expected chip gain.

Here is the current game state:
{state_str}

Available actions:
{', '.join(f'{action.name}: {desc}' for action, desc in available_actions.items())}

Always calculate pot odds versus hand strength and position. Consider stack-to-pot ratios and remaining streets.

IMPORTANT: You must respond with ONLY a single JSON object on one line, with no additional text or explanation.
The JSON must have exactly this format:
{{"action": "<bet/call/raise/check/fold>", "amount": int}}

Your response:"""
        
        try:
            # Get the response from the LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a poker player making decisions in a Texas Hold'em game. You must respond with ONLY a single JSON object containing the action and amount, with no additional text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=50
            )
            
            # Get the response content and clean it
            content = response.choices[0].message.content.strip()
            
            # Try to find JSON in the response if there's additional text
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                content = content[json_start:json_end]
            
            # Parse the JSON response
            try:
                action_json = json.loads(content)
                
                # Validate required fields
                if "action" not in action_json:
                    print("Error: Missing 'action' field in LLM response")
                    return ActionType.FOLD, None, None
                    
                action_str = action_json["action"].upper()
                amount = action_json.get("amount")
                
                # Convert action string to ActionType enum
                try:
                    action_type = ActionType[action_str]
                except KeyError:
                    print(f"Error: Invalid action type '{action_str}' in response")
                    return ActionType.FOLD, None, None
                    
                # Validate the action
                if action_type not in available_actions:
                    print(f"Error: Action '{action_type}' not available")
                    return ActionType.FOLD, None, None
                    
                # Format processed response as a simple string
                processed_response = action_type.name
                if amount is not None:
                    processed_response += f" {amount}"
                    
                # Print the processed response for debugging
                print(f"Processed action: {processed_response}")
                
                # Return the action
                return action_type, amount, None
                
            except json.JSONDecodeError as e:
                print(f"Error parsing LLM response as JSON: {str(e)}")
                print(f"Raw response: {content}")
                return ActionType.FOLD, None, None
                
        except Exception as e:
            print(f"Error getting action from LLM: {str(e)}")
            return ActionType.FOLD, None, None 
