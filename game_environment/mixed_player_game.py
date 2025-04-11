"""
Mixed player game implementation for Texas Hold'em poker.
This module provides a game where some players are controlled by LLMs and others are human-controlled.
"""

import os
import time
from typing import List, Dict, Optional, Tuple, Set, Union
from dotenv import load_dotenv
from texasholdem.game.game import TexasHoldEm
from texasholdem.gui.text_gui import TextGUI
from texasholdem.game.action_type import ActionType
from game_environment.llm_agent import LLMAgent


class MixedPlayerGame:
    """
    A Texas Hold'em game where some players are controlled by LLMs and others are human-controlled.
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
        Initialize the mixed player game.
        
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
        
        self.game = TexasHoldEm(buyin=buyin, big_blind=big_blind, small_blind=small_blind, max_players=max_players)
        self.gui = TextGUI(game=self.game)
        
        # Set up AI players
        if llm_player_ids is None:
            llm_player_ids = [0, 1]  # Default to players 0 and 1 being LLM-controlled
        
        self.llm_player_ids = set(llm_player_ids)
        self.human_player_ids = set(range(max_players)) - self.llm_player_ids
        
        # Initialize AI agents
        self.ai_agents = {}
        
        # Initialize LLM agents
        for player_id in self.llm_player_ids:
            self.ai_agents[player_id] = LLMAgent(model=openai_model, api_key=openai_api_key)
    
    def _is_ai_player(self, player_id: int) -> bool:
        """
        Check if a player is controlled by AI.
        
        Args:
            player_id: The ID of the player to check
            
        Returns:
            True if the player is controlled by AI, False otherwise
        """
        return player_id in self.llm_player_ids
    
    def _get_ai_action(self, player_id: int) -> Tuple[ActionType, Optional[int]]:
        """
        Get the action from an AI player.
        
        Args:
            player_id: The ID of the AI player
            
        Returns:
            A tuple of (action_type, total) where total is the amount to raise to (if applicable)
        """
        if player_id not in self.llm_player_ids:
            raise ValueError(f"Player {player_id} is not an LLM player")
            
        agent = self.ai_agents[player_id]
        return agent.get_action(self.game, player_id)
    
    def _get_human_action(self) -> Tuple[ActionType, Optional[int]]:
        """
        Get the action from a human player.
        
        Returns:
            A tuple of (action_type, total) where total is the amount to raise to (if applicable)
        """
        # Use the GUI to get the action from the human player
        self.gui.run_step()
        
        # The action is already taken by the GUI, so we just return None
        return None, None
    
    def run_game(self):
        """
        Run the game until it's over.
        """
        error_message = None
        try:
            while self.game.is_game_running():
                self.game.start_hand()
                
                while self.game.is_hand_running():
                    current_player = self.game.current_player
                    
                    if self._is_ai_player(current_player):
                        # Get action from AI
                        action_type, total = self._get_ai_action(current_player)
                        
                        # Take the action
                        if action_type == ActionType.RAISE and total is not None:
                            self.game.take_action(action_type, total=total)
                        else:
                            self.game.take_action(action_type)
                    else:
                        # Get action from human
                        self._get_human_action()
                
                # Export and replay the hand history
                pgn_path = self.game.export_history('./data/pgns')
                json_path = self.game.hand_history.export_history_json('./data/json')
                print(f"\nExported hand history to:")
                print(f"PGN: {pgn_path}")
                print(f"JSON: {json_path}")
                self.gui.replay_history(pgn_path)
                
                # Ask if the game should continue
                time.sleep(10)
                break
                
            
            print("Game over!")
            
        except Exception as e:
            # Save the error message
            error_message = f"\nError occurred: {str(e)}"
        else:
            # No error occurred
            error_message = None
        finally:
            # Always clean up the curses session
            self.gui.hide()
            # Reset the terminal
            os.system('reset')
            
            # Display the error message after cleanup if there was one
            if error_message:
                print(error_message)


if __name__ == "__main__":
    # Create a mixed player game with 6 players, where players 0 and 1 are LLM-controlled
    game = MixedPlayerGame(
        buyin=500,
        big_blind=5,
        small_blind=2,
        max_players=6,
        llm_player_ids=[0, 1],
        openai_model="gpt-4"
    )
    
    # Run the game
    game.run_game() 