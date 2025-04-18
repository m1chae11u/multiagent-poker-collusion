"""
Preflop solver implementation for Texas Hold'em poker.
This module provides functionality to load and use GTO preflop ranges.
"""

import os
import random
from typing import Dict, List, Tuple, Optional
from texasholdem.game.game import TexasHoldEm
from texasholdem.game.action_type import ActionType
from texasholdem.card.card import Card

class PreflopSolver:
    def __init__(self, ranges_dir: str = "cash6m_100bb_nl50_gto_gto/ranges"):
        """
        Initialize the preflop solver with the ranges directory.
        
        Args:
            ranges_dir: Directory containing the preflop ranges
        """
        self.ranges_dir = ranges_dir
        self.ranges_cache = {}  # Cache for loaded ranges
        
    def _get_hand_key(self, cards: List[Card]) -> str:
        """
        Convert two cards to a hand key (e.g., "AKs" for Ace-King suited)
        
        Args:
            cards: List of two cards
            
        Returns:
            Hand key string
        """
        if len(cards) != 2:
            raise ValueError("Must provide exactly 2 cards")
            
        # Sort cards by rank in descending order
        cards = sorted(cards, key=lambda c: c.rank, reverse=True)
        
        # Get ranks
        rank1 = Card.STR_RANKS[cards[0].rank]
        rank2 = Card.STR_RANKS[cards[1].rank]
        
        # Check if suited
        suited = cards[0].suit == cards[1].suit
        
        # Format hand key
        if rank1 == rank2:
            return f"{rank1}{rank1}"  # Pocket pair
        else:
            return f"{rank1}{rank2}{'s' if suited else 'o'}"  # Suited or offsuit
            
    def _load_range(self, position: str, bet_size: str, opponent: str, action: str) -> Dict[str, float]:
        """
        Load a range file from the ranges directory.
        
        Args:
            position: Player position (BTN, CO, HJ, SB, UTG)
            bet_size: Bet size (2.5bb, allin)
            opponent: Opponent position (BB, SB)
            action: Action type (call, allin)
            
        Returns:
            Dictionary mapping hand keys to frequencies
        """
        # Map position to correct bet size directory
        bet_size_map = {
            "BTN": "2.5bb",
            "CO": "2.3bb",
            "HJ": "2bb",
            "UTG": "2bb",
            "SB": "3bb"
        }
        
        # Get the correct bet size for this position
        actual_bet_size = bet_size_map.get(position, bet_size)
        
        # Construct the path based on the action type
        if action == "call":
            # For call action: ranges/POSITION/bet_size/opponent/call/opponent.txt
            path = os.path.join(self.ranges_dir, position, actual_bet_size, opponent, "call", f"{opponent}.txt")
        else:  # allin
            # For allin action: ranges/POSITION/allin/opponent/call/opponent.txt
            path = os.path.join(self.ranges_dir, position, "allin", opponent, "call", f"{opponent}.txt")
        
        # Check cache first
        if path in self.ranges_cache:
            return self.ranges_cache[path]
        
        # Load and parse range file
        try:
            with open(path, 'r') as f:
                content = f.read().strip()
                # Parse comma-separated hand frequencies
                range_dict = {}
                for hand_freq in content.split(','):
                    hand, freq = hand_freq.split(':')
                    range_dict[hand] = float(freq)
                self.ranges_cache[path] = range_dict
                return range_dict
        except FileNotFoundError:
            print(f"Warning: Range file not found at {path}")
            # If file not found, return empty range
            return {}
            
    def get_action(self, game: TexasHoldEm, player_id: int) -> Tuple[ActionType, Optional[int], Optional[str]]:
        """
        Get the next action based on GTO preflop ranges.
        
        Args:
            game: The Texas Hold'em game
            player_id: The ID of the player making the decision
            
        Returns:
            A tuple of (action_type, total, reason) where total is the amount to raise to (if applicable)
            and reason is the explanation for the action
        """
        # Get player's hand
        hand = game.get_hand(player_id)
        hand_key = self._get_hand_key(hand)
        
        # Determine position based on player index and button position
        # In a 6-max game, positions are: BTN, SB, BB, UTG, HJ, CO
        button_pos = game.btn_loc
        num_players = len(game.players)
        
        # Calculate relative position from button
        relative_pos = (player_id - button_pos) % num_players
        
        # Map relative position to position name
        position_map = {
            0: "BTN",  # Button
            1: "SB",   # Small Blind
            2: "BB",   # Big Blind
            3: "UTG",  # Under the Gun
            4: "HJ",   # Hijack
            5: "CO"    # Cut-off
        }
        
        # Get position key, default to BTN if unknown
        position_key = position_map.get(relative_pos, "BTN")
        
        # Get available moves
        available_moves = game.get_available_moves()
        
        # Determine action based on available moves and ranges
        if ActionType.CALL in available_moves.action_types:
            # Load call range
            call_range = self._load_range(position_key, "2.5bb", "BB", "call")
            # Get the probability for this hand
            call_prob = call_range.get(hand_key, 0.0)
            
            # Make decision based on probability
            if random.random() < call_prob:
                return ActionType.CALL, None, f"Preflop GTO strategy: {hand_key} has {call_prob:.2%} probability to call from {position_key}"
                
        if ActionType.RAISE in available_moves.action_types:
            # Load raise range
            raise_range = self._load_range(position_key, "2.5bb", "BB", "allin")
            # Get the probability for this hand
            raise_prob = raise_range.get(hand_key, 0.0)
            
            # Make decision based on probability
            if random.random() < raise_prob:
                # Get minimum raise amount and convert to total amount
                min_raise_value = game.min_raise()
                if min_raise_value <= 0:
                    min_raise_value = game.big_blind  # Use big blind size as minimum raise
                
                # Convert the raise value to a total amount
                min_raise_total = game.value_to_total(min_raise_value, player_id)
                
                return ActionType.RAISE, min_raise_total, f"Preflop GTO strategy: {hand_key} has {raise_prob:.2%} probability to raise from {position_key}"
                
        # Default to check if available
        if ActionType.CHECK in available_moves.action_types:
            # Load call range to get the probability
            call_range = self._load_range(position_key, "2.5bb", "BB", "call")
            call_prob = call_range.get(hand_key, 0.0)
            return ActionType.CHECK, None, f"Preflop GTO strategy: {hand_key} has {call_prob:.2%} probability to call/raise from {position_key}, checking instead"
            
        # Default to fold if no other options
        # Load call range to get the probability
        call_range = self._load_range(position_key, "2.5bb", "BB", "call")
        call_prob = call_range.get(hand_key, 0.0)
        return ActionType.FOLD, None, f"Preflop GTO strategy: {hand_key} has {call_prob:.2%} probability to call/raise from {position_key}, folding instead" 