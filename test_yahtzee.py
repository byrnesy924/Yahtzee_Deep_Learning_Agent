# This script is designed to test the Yahtzee game code
import pytest
from Yahtzee import Yahtzee


if __name__ == "__main__":
    test_game = Yahtzee(player_type="Model")

    # pick_score - highly coupled and complex, eek
    def test_pick_score_singles(yahtzee_game: Yahtzee):
        """Test the singles pick score. Assume represents all ones - sixes and check code is mirror"""        
        
        # ones - sixes
        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.dice_saved = [1, 1, 1, 1, 1]
        yahtzee_game.pick_score("ones")
        assert yahtzee_game.ones == 5

        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.dice_saved = [1, 1, 1, 2, 2]
        yahtzee_game.pick_score("ones")
        assert yahtzee_game.ones == 3

        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.dice_saved = [1, 1]
        yahtzee_game.pick_score("ones")
        assert yahtzee_game.ones == 2

        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.dice_saved = [2, 2]
        yahtzee_game.pick_score("ones")
        assert yahtzee_game.ones == 0

        # test case - cant choose ones, already chosen
        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.chosen_scores = ["ones", "twos"]
        yahtzee_game.dice_saved = [1, 1]
        yahtzee_game.pick_score("ones")
        assert yahtzee_game.ones == 0

    def test_pick_score_bottom_half(yahtzee_game: Yahtzee):
        """Test the bottom poker style scores"""
        # three of a kind
        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.dice_saved = [2, 2, 2]
        yahtzee_game.pick_score("three_of_a_kind")
        assert yahtzee_game.three_of_a_kind == 6

        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.dice_saved = [2, 1, 5, 5, 5]
        yahtzee_game.pick_score("three_of_a_kind")
        assert yahtzee_game.three_of_a_kind == 18

        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.chosen_scores = ["three_of_a_kind", "twos"]
        yahtzee_game.dice_saved = [1, 1, 6, 6, 6]
        yahtzee_game.pick_score("three_of_a_kind")
        assert yahtzee_game.three_of_a_kind == 0

        # full house
        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.dice_saved = [2, 2, 5, 5, 5]
        yahtzee_game.pick_score("full_house")
        assert yahtzee_game.full_house == 25

        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.dice_saved = [2, 2, 4, 5, 5]
        yahtzee_game.pick_score("full_house")
        assert yahtzee_game.full_house == 0

        # straights - start with small
        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.dice_saved = [1, 2, 4, 5]
        yahtzee_game.pick_score("small_straight")
        assert yahtzee_game.small_straight == 0

        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.dice_saved = [1, 2, 3, 4, 5]
        yahtzee_game.pick_score("small_straight")
        assert yahtzee_game.small_straight == 30

        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.dice_saved = [3, 4, 5, 6, 6]
        yahtzee_game.pick_score("small_straight")
        assert yahtzee_game.small_straight == 30

        # Large straight
        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.dice_saved = [1, 2, 3, 4, 5]
        yahtzee_game.pick_score("large_straight")
        assert yahtzee_game.large_straight == 30

        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.dice_saved = [1, 2, 4, 5]
        yahtzee_game.pick_score("large_straight")
        assert yahtzee_game.large_straight == 0

        # test yahtzee
        yahtzee_game.reset_game()
        yahtzee_game.sub_turn = 3
        yahtzee_game.dice_saved = [5, 5, 5, 5, 5]
        yahtzee_game.pick_score("yahtzee")
        assert yahtzee_game.yahtzee == 50

        # and yahtzee bonus
        yahtzee_game.pick_score("yahtzee")
        assert yahtzee_game.yahtzee_bonus == 100

    # turn
    # reset games
    def test_reset_game(yahtzee_game: Yahtzee):
        yahtzee_game.reset_game()
        assert (
             yahtzee_game.turn_number,
             yahtzee_game.sub_turn,
             yahtzee_game.dice_saved,
             yahtzee_game.chosen_scores,
             yahtzee_game.get_bonus
        ) == (
            1,
            1,
            list(),
            list(),
            False
        )
    test_reset_game(test_game)
    test_pick_score_singles(test_game)
    test_pick_score_bottom_half(test_game)

    # Do the below last
    # pick ones - sixes but set test_game.dice_saved before hand

    # pick_three_of_a_kind, four of a kind etc.
    print("Tests passed")


