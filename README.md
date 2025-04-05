
## Setup

1. Clone this repository
2. (Optional) Create your own virtual environment
   ``` python -m venv venv
      source venv/bin/activate    # On Windows, use: venv\Scripts\activate
```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Create a `.env` file based on the `example.env` file:
   ```
   cp example.env .env
   ```
6. Edit the `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Running the Game

To run the game with default settings (6 players, where players 0 and 1 are AI-controlled):

```python
from game_environment.mixed_player_game import MixedPlayerGame

# Create a mixed player game
game = MixedPlayerGame(
    buyin=500,
    big_blind=5,
    small_blind=2,
    max_players=6,
    ai_player_ids=[0, 1]  # Players 0 and 1 are AI-controlled
)

# Run the game
game.run_game()
```

Or use the command-line script:

```bash
python run_game.py
```

### Customizing the Game

You can customize the game by changing the parameters when initializing the `MixedPlayerGame` class:

- `buyin`: The amount of chips each player starts with (default: 500)
- `big_blind`: The big blind amount (default: 5)
- `small_blind`: The small blind amount (default: 2)
- `max_players`: The maximum number of players (default: 6)
- `ai_player_ids`: The IDs of players controlled by AI (default: [0, 1])
- `openai_model`: The OpenAI model to use (default: value in .env file)
- `openai_api_key`: The OpenAI API key (default: value in .env file)

You can also customize the game using command-line arguments:

```bash
python run_game.py --max-players 8 --ai-players 0,1,2 --model gpt-3.5-turbo
```

### How to Play

1. When it's your turn, you'll be prompted to enter an action. Note: make sure to press backspace a couple times before inputting action as currently an issue with ui.
2. Available actions are:
   - `check`: Pass the action
   - `call`: Match the current bet
   - `raise <amount>`: Increase the bet to the specified amount
   - `fold`: Give up the hand
   - `all_in`: Bet all your chips

3. After each hand, you'll be asked if you want to continue playing.

## Project Structure

- `game_environment/openai_agent.py`: Implementation of the OpenAI agent
- `game_environment/mixed_player_game.py`: Implementation of the mixed player game
- `run_game.py`: Command-line script to run the game
- `requirements.txt`: List of required dependencies
- `example.env`: Example environment variables file
- `.env`: Environment variables file (create from example.env)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
