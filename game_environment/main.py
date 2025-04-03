#Game enviroment for Poker
#Finished Version 4.0
#Ready for testing

#VOCAB NEEDED TO UNDERSTAND THE GAME PLEASE LOOK AT THE VOCAB
#FLOP: The first 3 cards of the deck
#Turn: The 4th card of the deck
#River: The 5th card of the deck

# Code already finished just copy and pasted from my scratchboard

import random
from collections import Counter

# Card setup
suits = ['♠', '♥', '♦', '♣']
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
rank_values = {rank: i for i, rank in enumerate(ranks, 2)}

# Poker hand rankings
hand_ranking = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind",
    "Straight Flush", "Royal Flush"
]

def create_deck():
    return [rank + suit for suit in suits for rank in ranks]

def shuffle_deck(deck):
    random.shuffle(deck)

def deal_cards(deck, num):
    return [deck.pop() for _ in range(num)]

# Hand evaluation
def evaluate_hand(cards):
    values = sorted([rank_values[c[:-1]] for c in cards], reverse=True)
    suits_ = [c[-1] for c in cards]
    ranks_ = [c[:-1] for c in cards]

    is_flush = len(set(suits_)) == 1
    is_straight = all(values[i] - 1 == values[i+1] for i in range(len(values)-1))

    count = Counter(ranks_)
    most_common = count.most_common()
    counts = sorted([v for k, v in most_common], reverse=True)

    if is_flush and is_straight:
        if values[0] == 14:
            return (9, values)  # Royal Flush
        return (8, values)  # Straight Flush
    elif counts[0] == 4:
        return (7, values)  # Four of a Kind
    elif counts[0] == 3 and counts[1] == 2:
        return (6, values)  # Full House
    elif is_flush:
        return (5, values)
    elif is_straight:
        return (4, values)
    elif counts[0] == 3:
        return (3, values)
    elif counts[0] == 2 and counts[1] == 2:
        return (2, values)
    elif counts[0] == 2:
        return (1, values)
    else:
        return (0, values)  # High Card

# Player class
class Player:
    def __init__(self, name, chips=1000):
        self.name = name
        self.chips = chips
        self.hand = []
        self.in_game = True

    def bet(self, amount):
        bet_amount = min(amount, self.chips)
        self.chips -= bet_amount
        return bet_amount

    def show_hand(self):
        return ' '.join(self.hand)

# Game environment
class PokerGame:
    def __init__(self):
        self.deck = create_deck()
        shuffle_deck(self.deck)
        self.pot = 0
        self.community_cards = []

        self.player = Player("You")
        self.bot = Player("Bot")
        self.players = [self.player, self.bot]

    def deal_hole_cards(self):
        for player in self.players:
            player.hand = deal_cards(self.deck, 2)

    def deal_flop(self):
        self.deck.pop()
        self.community_cards += deal_cards(self.deck, 3)

    def deal_turn_or_river(self):
        self.deck.pop()
        self.community_cards += deal_cards(self.deck, 1)

    def show_community_cards(self):
        return ' '.join(self.community_cards)

    def betting_round(self):
        print("\n--- Betting Round ---")

        # Player decision
        if self.player.in_game:
            print(f"{self.player.name} chips: {self.player.chips}")
            action = input("Choose action (check / bet / fold): ").strip().lower()
            if action == "fold":
                self.player.in_game = False
            elif action == "bet":
                amount = int(input("Enter bet amount: "))
                self.pot += self.player.bet(amount)
                print(f"You bet {amount} chips.")
            else:
                print("You checked.")

        # Bot decision
        if self.bot.in_game:
            strength, _ = evaluate_hand(self.bot.hand + self.community_cards)
            if strength >= 2:
                amount = 50
                self.pot += self.bot.bet(amount)
                print("Bot bets 50 chips.")
            else:
                print("Bot checks.")

    def play_game(self):
        print("\n--- Starting Poker Game ---")
        self.deal_hole_cards()

        print(f"Your hand: {self.player.show_hand()}")

        self.betting_round()
        self.deal_flop()
        print(f"Flop: {self.show_community_cards()}")
        self.betting_round()

        self.deal_turn_or_river()
        print(f"Turn: {self.show_community_cards()}")
        self.betting_round()

        self.deal_turn_or_river()
        print(f"River: {self.show_community_cards()}")
        self.betting_round()

        if not self.player.in_game:
            print("You folded. Bot wins!")
            return
        elif not self.bot.in_game:
            print("Bot folded. You win!")
            return

        # Evaluate hands
        player_best = evaluate_hand(self.player.hand + self.community_cards)
        bot_best = evaluate_hand(self.bot.hand + self.community_cards)

        print(f"\nYour final hand: {self.player.show_hand()} -> {hand_ranking[player_best[0]]}")
        print(f"Bot final hand: {self.bot.show_hand()} -> {hand_ranking[bot_best[0]]}")

        if player_best > bot_best:
            print("You win the pot!")
            self.player.chips += self.pot
        elif player_best < bot_best:
            print("Bot wins the pot!")
            self.bot.chips += self.pot
        else:
            print("It's a tie! Splitting pot.")
            split = self.pot // 2
            self.player.chips += split
            self.bot.chips += split

        print(f"Pot was: {self.pot} chips")
        print(f"Your chips: {self.player.chips}")
        print(f"Bot chips: {self.bot.chips}")

# Run the game
if __name__ == "__main__":
    game = PokerGame()
    while True:
        game.play_game()
        again = input("Play again? (y/n): ").strip().lower()
        if again != 'y':
            break
