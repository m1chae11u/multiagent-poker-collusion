"""
Two LLM game implementation for Texas Hold'em poker.
This module provides a game where two players are controlled by LLMs and the rest are CFR agents using robopoker.
"""

import os
import time
from typing import List, Dict, Optional, Tuple, Set, Union
from dotenv import load_dotenv
from texasholdem.game.game import TexasHoldEm
from texasholdem.game.action_type import ActionType
from texasholdem.game.hand_phase import HandPhase
from game_environment.llm_agent import LLMAgent
from game_environment.robopoker_agent import RobopokerAgent


class TwoLLMGame:
    """
    A Texas Hold'em game where two players are controlled by LLMs and the rest are CFR agents using robopoker.
    """
    
    def __init__(
        self,
        buyin: int = 500,
        big_blind: int = 5,
        small_blind: int = 2,
        max_players: int = 6,
        llm_player_ids: Optional[List[int]] = None,
        openai_model: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the two LLM game.
        
        Args:
            buyin: The amount of chips each player starts with
            big_blind: The big blind amount
            small_blind: The small blind amount
            max_players: The maximum number of players
            llm_player_ids: The IDs of players controlled by LLM. If None, players 0 and 1 will be LLM-controlled.
            openai_model: The model name to use. If None, will try to get from .env file
            openai_api_key: The API key. If None, will try to get from .env file
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Ensure robopoker binary exists
        if not os.path.exists("robopoker/target/release/robopoker"):
            raise RuntimeError("robopoker binary not found. Please build it first.")
        
        self.game = TexasHoldEm(buyin=buyin, big_blind=big_blind, small_blind=small_blind, max_players=max_players)
        
        # Set up AI players
        if llm_player_ids is None:
            llm_player_ids = [0, 1]  # Default to players 0 and 1 being LLM-controlled
        
        self.llm_player_ids = set(llm_player_ids)
        self.cfr_player_ids = set(range(max_players)) - self.llm_player_ids
        self.ai_player_ids = self.llm_player_ids.union(self.cfr_player_ids)
        
        # Initialize AI agents
        self.ai_agents = {}
        
        # Initialize LLM agents
        for player_id in self.llm_player_ids:
            self.ai_agents[player_id] = LLMAgent(model=openai_model, api_key=openai_api_key)
        
        # Initialize Robopoker agents
        for player_id in self.cfr_player_ids:
            self.ai_agents[player_id] = RobopokerAgent()
        
        # Track starting chips for each phase
        self.phase_start_chips = {}
    
    def _is_ai_player(self, player_id: int) -> bool:
        """
        Check if a player is controlled by AI.
        
        Args:
            player_id: The ID of the player to check
            
        Returns:
            True if the player is controlled by AI, False otherwise
        """
        return player_id in self.ai_player_ids
    
    def _get_ai_action(self, player_id: int) -> Tuple[ActionType, Optional[int], Optional[str]]:
        """
        Get the action from an AI player.
        
        Args:
            player_id: The ID of the AI player
            
        Returns:
            A tuple of (action_type, total, reason) where total is the amount to raise to (if applicable)
            and reason is the explanation for the action (if available)
        """
        agent = self.ai_agents[player_id]
        return agent.get_action(self.game, player_id)
    
    def _print_game_state(self):
        """Print the current state of the game."""
        print("\n=== Current Game State ===")
        print(f"Hand Phase: {self.game.hand_phase.name}")
        print(f"Board: {[str(card) for card in self.game.board]}")
        print(f"Pot: {self.game._get_last_pot().get_total_amount()}")
        print("\nPlayers:")
        for i, player in enumerate(self.game.players):
            player_type = "LLM" if i in self.llm_player_ids else "Robopoker"
            print(f"Player {i} ({player_type}): {player.chips} chips")
            if i in self.game.hands:
                print(f"  Hand: {[str(card) for card in self.game.hands[i]]}")
        print("=======================\n")
    
    def _print_action(self, player_id: int, action_type: ActionType, total: Optional[int] = None, reason: Optional[str] = None):
        """Print a player's action."""
        player_type = "LLM" if player_id in self.llm_player_ids else "Robopoker"
        action_str = action_type.name
        if total is not None:
            action_str += f" to {total}"
        print(f"Player {player_id} ({player_type}): {action_str}")
        if reason:
            print(f"  Reason: {reason}")
    
    def _print_round_start(self, phase: HandPhase):
        """Print the start of a betting round."""
        print(f"\n{'='*20} {phase.name} {'='*20}")
        if phase == HandPhase.FLOP:
            print("Dealing the flop...")
        elif phase == HandPhase.TURN:
            print("Dealing the turn...")
        elif phase == HandPhase.RIVER:
            print("Dealing the river...")
        
        # Save starting chips for this phase
        self.phase_start_chips = {i: player.chips for i, player in enumerate(self.game.players)}
        print("\nStarting chips for this round:")
        for i, chips in self.phase_start_chips.items():
            player_type = "LLM" if i in self.llm_player_ids else "Robopoker"
            print(f"Player {i} ({player_type}): {chips} chips")
        
        self._print_game_state()
    
    def _print_round_end(self, phase: HandPhase):
        """Print the end of a betting round."""
        print(f"\n{'='*20} {phase.name} Summary {'='*20}")
        print("Chip changes in this round:")
        for i, player in enumerate(self.game.players):
            player_type = "LLM" if i in self.llm_player_ids else "Robopoker"
            start_chips = self.phase_start_chips[i]
            end_chips = player.chips
            change = end_chips - start_chips
            change_str = f"+{change}" if change > 0 else str(change)
            print(f"Player {i} ({player_type}): {start_chips} -> {end_chips} ({change_str})")
        print(f"Total pot: {self.game._get_last_pot().get_total_amount()}")
        print("="*60)
    
    def run_game(self):
        """
        Run the game until it's over.
        """
        error_message = None
        try:
            while self.game.is_game_running():
                self.game.start_hand()
                print("\n=== Starting New Hand ===")
                print(f"Button is at Player {self.game.btn_loc}")
                print(f"Small blind ({self.game.small_blind}) posted by Player {self.game.sb_loc}")
                print(f"Big blind ({self.game.big_blind}) posted by Player {self.game.bb_loc}")
                
                # Save starting chips for preflop
                self.phase_start_chips = {i: player.chips for i, player in enumerate(self.game.players)}
                print("\nStarting chips for this hand:")
                for i, chips in self.phase_start_chips.items():
                    player_type = "LLM" if i in self.llm_player_ids else "Robopoker"
                    print(f"Player {i} ({player_type}): {chips} chips")
                
                self._print_game_state()
                
                # Track the current phase to detect phase changes
                current_phase = self.game.hand_phase
                
                while self.game.is_hand_running():
                    # Check if we've moved to a new phase
                    if self.game.hand_phase != current_phase:
                        # Print summary of previous phase
                        self._print_round_end(current_phase)
                        current_phase = self.game.hand_phase
                        self._print_round_start(current_phase)
                    
                    current_player = self.game.current_player
                    
                    if self._is_ai_player(current_player):
                        # Get action from AI
                        action_type, total, reason = self._get_ai_action(current_player)
                        
                        # Print the action
                        self._print_action(current_player, action_type, total, reason)
                        
                        # Take the action
                        if action_type == ActionType.RAISE and total is not None:
                            self.game.take_action(action_type, total=total)
                        else:
                            self.game.take_action(action_type)
                
                # Print summary of final phase
                self._print_round_end(current_phase)
                
                # Export hand history
                pgn_path = self.game.export_history('./pgns')
                json_path = self.game.hand_history.export_history_json('./pgns')
                print(f"\nExported hand history to:")
                print(f"PGN: {pgn_path}")
                print(f"JSON: {json_path}")
                
                # Print final hand state
                print("\n=== Hand Complete ===")
                self._print_game_state()
                
                # Wait a bit before next hand
                time.sleep(2)
            
            print("\nGame over!")
            
        except Exception as e:
            # Save the error message
            error_message = f"\nError occurred: {str(e)}"
        else:
            # No error occurred
            error_message = None
        finally:
            # Display the error message if there was one
            if error_message:
                print(error_message)


if __name__ == "__main__":
    # Create a two LLM game with 6 players, where players 0 and 1 are LLM-controlled and the rest are Robopoker-controlled
    game = TwoLLMGame(
        buyin=500,
        big_blind=5,
        small_blind=2,
        max_players=6,
        llm_player_ids=[0, 1],
        openai_model="gpt-4"
    )
    
    # Run the game
    game.run_game()