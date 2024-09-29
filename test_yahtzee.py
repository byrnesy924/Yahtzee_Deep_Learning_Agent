# This script is designed to test the Yahtzee game code
import pytest
from Yahtzee import Yahtzee


if __name__ == "__main__":
    test_game = Yahtzee(player_type="Model")

    # pick_score - highly coupled and complex, eek
    # turn
    # reset games
    def test_reset_game(yahtzee_game: Yahtzee):
        yahtzee_game.reset_game()
        assert (
             yahtzee_game.turn_number,
             yahtzee_game.sub_turn,
             yahtzee_game.dice_saved,
             yahtzee_game.chosen_scores,
             yahtzee_game.third_roll,
             yahtzee_game.second_roll,
             yahtzee_game.first_roll,
             yahtzee_game.get_bonus
        ) == (
            1,
            2,
            list(),
            list(),
            {"one": 0, "two": 0, "three": 0, "four": 0, "five": 0},
            {"one": 0, "two": 0, "three": 0, "four": 0, "five": 0},
            {"one": 0, "two": 0, "three": 0, "four": 0, "five": 0},
            False
        )
    test_reset_game(test_game)

    # Do the below last
    # pick ones - sixes but set test_game.dice_saved before hand

    # pick_three_of_a_kind, four of a kind etc.

