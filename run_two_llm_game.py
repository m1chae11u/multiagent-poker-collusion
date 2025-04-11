#!/usr/bin/env python3
"""
Script to run the two LLM poker game.
"""

import os
import argparse
from game_environment.two_llm_game import TwoLLMGame


def main():
    """
    Main function to run the game.
    """
    parser = argparse.ArgumentParser(description="Run a two LLM poker game")
    parser.add_argument("--buyin", type=int, default=500, help="The amount of chips each player starts with")
    parser.add_argument("--big-blind", type=int, default=5, help="The big blind amount")
    parser.add_argument("--small-blind", type=int, default=2, help="The small blind amount")
    parser.add_argument("--max-players", type=int, default=6, help="The maximum number of players")
    parser.add_argument("--llm-players", type=str, default="0,1", help="Comma-separated list of LLM player IDs")
    parser.add_argument("--model", type=str, default="gpt-4", help="The OpenAI model to use")
    parser.add_argument("--api-key", type=str, default=None, help="The OpenAI API key")
    
    args = parser.parse_args()
    
    # Parse player IDs
    llm_player_ids = [int(id_str) for id_str in args.llm_players.split(",")] if args.llm_players else []
    
    # Create the game
    game = TwoLLMGame(
        buyin=args.buyin,
        big_blind=args.big_blind,
        small_blind=args.small_blind,
        max_players=args.max_players,
        llm_player_ids=llm_player_ids,
        openai_model=args.model,
        openai_api_key=args.api_key
    )
    
    # Run the game
    game.run_game()


if __name__ == "__main__":
    main() 