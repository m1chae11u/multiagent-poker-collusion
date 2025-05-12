import postflop_solver
from postflop_solver import SolverState, get_optimal_action
from texasholdem.game.game import TexasHoldEm
from texasholdem.game.action_type import ActionType

def main():
    # Create a test state for a 6-player table
    # Player positions: 0=UTG, 1=MP, 2=CO, 3=BTN, 4=SB, 5=BB
    board_cards = ["Ah", "Kh", "Qh", "2c", "3d"]  # Flop + Turn + River (part of royal flush)
    hole_cards = ["Jh", "Th"]  # Player's hole cards (completing royal flush)
    pot_size = 1000  # Current pot size
    stack_sizes = [1000, 1000, 1000, 1000, 1000, 1000]  # All players start with 1000 chips
    position = 3  # Player is in position (BTN) - last to act
    current_bet = 200  # Current bet amount
    must_call = True  # Whether player must call
    valid_actions = ["check", "call", "raise", "fold", "all_in"]  # Available actions
    betting_round = "RIVER"  # Current betting round

    # Create a solver state using the imported SolverState class
    state = SolverState(
        board_cards=board_cards,
        hole_cards=hole_cards,
        pot_size=pot_size,
        stack_sizes=stack_sizes,
        position=position,
        current_bet=current_bet,
        must_call=must_call,
        valid_actions=valid_actions,
        betting_round=betting_round
    )

    try:
        # Get the optimal action
        result = get_optimal_action(state)
        print("\nGame State:")
        print(f"Board: {' '.join(board_cards)}")
        print(f"Hole Cards: {' '.join(hole_cards)}")
        print(f"Hand: Royal Flush in Hearts")
        print(f"Position: {'OOP' if position == 0 else 'IP'} (position {position})")
        print(f"Pot Size: {pot_size}")
        print(f"Stack Sizes: {stack_sizes}")
        print(f"Current Bet: {current_bet}")
        print(f"Must Call: {must_call}")
        print(f"Betting Round: {betting_round}")
        print("\nSolver's Decision:")
        print(f"Action: {result.action}")
        print(f"Amount: {result.amount}")
        print(f"Reason: {result.reason}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

# Print available attributes
print("\nAvailable attributes in postflop_solver module:")
for attr in dir(postflop_solver):
    if not attr.startswith('_'):
        print(f"- {attr}") 