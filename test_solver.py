import postflop_solver
from postflop_solver import SolverState, get_optimal_action

def main():
    # Create a test state
    board_cards = ["Ah", "Kh", "Qh"]  # Flop cards
    hole_cards = ["Jh", "Th"]  # Player's hole cards
    pot_size = 100  # Current pot size
    stack_sizes = [900, 900]  # Remaining stack sizes for both players
    position = 1  # Player's position (0 for OOP, 1 for IP)
    current_bet = 75  # Current bet amount
    must_call = True  # Whether player must call
    valid_actions = ["check", "call", "raise", "fold", "all_in"]  # Available actions
    betting_round = "FLOP"  # Current betting round

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
        print(f"Optimal action: {result.action}")
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