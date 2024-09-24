# This script is designed to test the Yahtzee game code
from Yahtzee import Yahtzee



if __name__ == "__main__":
    test_game = Yahtzee(player_type="Model")

    # To test:
    # pick_dice
    # pick_score - highly coupled and complex, eek
    # turn
    # reset games

    # Do the below last
    # pick ones - sixes but set test_game.dice_saved before hand

    # pick_three_of_a_kind, four of a kind etc.

