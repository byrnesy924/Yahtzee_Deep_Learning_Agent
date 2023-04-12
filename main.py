import pandas as pd
import numpy as np
import random
import time

class Yahtzee:
    turn_number: int = 1  # Note starts from 1
    sub_turn: int = 1  # Note starts from 1

    # Singles
    ones: int = 0
    twos: int = 0
    threes: int = 0
    fours: int = 0
    fives: int = 0
    sixes: int = 0

    # Specials
    three_of_a_kind: int = 0
    four_of_a_kind: int = 0
    full_house: int = 0
    small_straight: int = 0
    large_straight: int = 0
    chance: int = 0
    yahtzee: int = 0
    yahtzee_bonus: int = 0
    singles_total: int = 0
    total_score: int = 0
    get_bonus: bool = False

    # Rolls
    empty_roll = {"one": 0, "two": 0, "three": 0, "four": 0, "five": 0}
    first_roll: dict = {"one": 0, "two": 0, "three": 0, "four": 0, "five": 0}
    second_roll: dict = {"one": 0, "two": 0, "three": 0, "four": 0, "five": 0}
    third_roll: dict = {"one": 0, "two": 0, "three": 0, "four": 0, "five": 0}
    dice_saved: list = []

    def __init__(self):
        return

    def roll_dice(self):
        numbers = [random.randint(1, 6) for i in range(15)]

        # commented to make super lightweight for training
        # if numbers.count(0) > 0:
        #     raise Exception("Dice roll contains 0. Random package screwed you buddy")

        self.first_roll, self.second_roll, self.third_roll = self.empty_roll, self.empty_roll, self.empty_roll

        for key in self.first_roll:
            self.first_roll[key] = numbers.pop(0)
        for key in self.second_roll:
            self.second_roll[key] = numbers.pop(0)
        for key in self.third_roll:
            self.third_roll[key] = numbers.pop()

    def pick_dice(self, dice_to_choose: dict):
        """This method will choose dice from the current roll, and then update future rolls appropriately
        :param dice_to_choose: a dictionary with the same keys as the roll dictionaries and Bool values

        """
        active_roll_mapper = {1: "first_roll", 2: "second_roll", 3: "third_roll"}
        update_roll_mapper = {1: ["second_roll", "third_roll"], 2: ["third_roll"], 3: None}
        active_roll_name = active_roll_mapper[self.sub_turn]
        active_roll = self.__getattribute__(active_roll_name)
        list_of_rolls_to_update = update_roll_mapper[self.sub_turn]

        # get the current roll
        for key in dice_to_choose:
            if dice_to_choose[key]:
                # Get the current roll
                self.dice_saved.append(active_roll[key])

                # eliminate the chosen dice from future rolls - very nested code is not great but not end of the world
                # if statement skips code block if its the last sub-roll on a turn (i.e. third roll of dice)
                if update_roll_mapper is not None:
                    # Go through the future dictionaries
                    for roll in list_of_rolls_to_update:
                        # Get the future dict
                        temp_dict = self.__getattribute__(roll)
                        # update to contain a 0
                        temp_dict[key] = 0
                        # reset it
                        self.__setattr__(roll, temp_dict)

    def pick_ones(self):
        return self.dice_saved.count(1)

    def pick_twos(self):
        return 2 * self.dice_saved.count(2)

    def pick_threes(self):
        return 3 * self.dice_saved.count(3)

    def pick_fours(self):
        return 4 * self.dice_saved.count(4)

    def pick_fives(self):
        return 5 * self.dice_saved.count(5)

    def pick_sixes(self):
        return 6 * self.dice_saved.count(6)

    def pick_three_of_a_kind(self):
        success = False
        for i in range(1, 7):
            success = self.dice_saved.count(i) >= 3 or success  # If the count is >= 3 or has ever been >= 3, then True

        if success:
            return sum(self.dice_saved)
        return 0

    def pick_four_of_a_kind(self):
        success = False
        for i in range(1, 7):
            success = self.dice_saved.count(i) >= 4 or success  # If the count is >= 4 or has ever been >= 3, then True

        if success:
            return sum(self.dice_saved)
        return 0

    def pick_full_house(self):
        """Logic: if theres a 2 of any kind and a 3 of any kind, Or a 4 and 1 or a 5 and 0,
        then the set only contains 2 dice and 5 total dice were chosen"""
        unique_dice_chosen = list(set(self.dice_saved))  # convert back to list to subscript it later in or statement
        if len(self.dice_saved) == 5 \
                and len(unique_dice_chosen) <= 2 \
                and (self.dice_saved.count(unique_dice_chosen[0]) == 3
                     or self.dice_saved.count(unique_dice_chosen[0]) == 3):
            return 25
        return 0

    def pick_small_straight(self):
        """Pick a 4 in a row straight. abuse that there is only 3 possibilities for this, and that sets are sorted"""
        unique_dice_chosen = list(set(self.dice_saved))
        if unique_dice_chosen[0:4] == [1, 2, 3, 4] or unique_dice_chosen[0:4] == [2, 3, 4, 5] \
                or unique_dice_chosen[0:4] == [3, 4, 5, 6]:
            return 30
        return 0

    def pick_large_straight(self):
        """Pick a 5 in a row straight. abuse that there is only 2 possibilities for this, and that sets are sorted"""
        unique_dice_chosen = list(set(self.dice_saved))
        if unique_dice_chosen[0:5] == [1, 2, 3, 4, 5] or unique_dice_chosen[0:5] == [2, 3, 4, 5, 6]:
            return 30
        return 0

    def pick_chance(self):
        return sum(self.dice_saved)

    def pick_yahtzee(self):
        if len(self.dice_saved) == 5 and len(set(self.dice_saved)) == 1:
            return 50
        return 0

    def check_yahtzee_bonus(self):
        if len(self.dice_saved) == 5 and len(
                set(self.dice_saved)) == 1 and self.yahtzee == 50 and self.yahtzee_bonus == 0:
            self.yahtzee_bonus = 100
        return

    def pick_score(self, pick: str):
        """TODO sure up how to implement this code with Tensor Flow
        wrapper for picking scores. Currently works on string input.
        """
        if self.sub_turn != 3:
            raise Exception("Tried to pick a score but its not the third roll")
        if pick == "ones":
            self.ones = self.pick_ones()
        if pick == "twos":
            self.twos = self.pick_twos()
        if pick == "threes":
            self.threes = self.pick_threes()
        if pick == "fours":
            self.fours = self.pick_fours()
        if pick == "fives":
            self.fives = self.pick_fives()
        if pick == "sixes":
            self.sixes = self.pick_sixes()
        if pick == "three_of_a_kind":
            self.three_of_a_kind = self.pick_three_of_a_kind()
        if pick == "four_of_a_kind":
            self.four_of_a_kind = self.pick_four_of_a_kind()
        if pick == "full_house":
            self.full_house = self.pick_full_house()
        if pick == "small_straight":
            self.small_straight = self.pick_small_straight()
        if pick == "large_straight":
            self.large_straight = self.pick_large_straight()
        if pick == "chance":
            self.chance = self.pick_chance()
        if pick == "yahtzee":
            self.yahtzee = self.pick_yahtzee()
        self.check_yahtzee_bonus()
        return pick

    def turn(self, player_input=False, choice=None):

        if self.sub_turn == 1:
            self.dice_saved = []
            self.roll_dice()
        if player_input:
            print("\nThis is the current dice roll. 0's mean they cannot be selected") #TODO make sure it cant do that
            print("Dice roll: \n")
            self.print_roll()
            print("\n")
            print("You will now be asked for each dice whether to keep it or not. 1 is keep.")
            die = ["one", "two", "three", "four", "five"]
            choice = {}
            for index, dice in enumerate(die):
                choice[dice] = int(input(f"Input a 0 or 1 for dice number {dice}"))
        else:
            # just pick the first dice roll
            if self.sub_turn == 1:
                choice = {"one": 1, "two": 1, "three": 1, "four": 1, "five": 1}
            else:
                choice = {"one": 0, "two": 0, "three": 0, "four": 0, "five": 0}

        # TODO need a way for the net to actually feed in this choice
        print("Your choice was: ", choice)
        self.pick_dice(choice)

        # If its the third dice roll i.e. sub turn 3, then pick a score!
        if self.sub_turn == 3:
            if player_input:
                # TODO make input incorporate the edge case of an incorrect input
                print("Your chosen dice are: \n")
                print(self.dice_saved)
                score_choice = input("""input a score choice. The available are: singles (e.g. ones), three_of_a_kind, 
                four_of_a_kind, full_house, small_straight, large_straight, and yahtzee.""")
            else:
                # created this temp function to just pick something randomly until the end
                # TODO: need a way to feed in this choice from the net
                score_choice = self.temp_randomly_pick_score()

            self.pick_score(score_choice)
            self.turn_number += 1
            self.sub_turn = 1
            print(f"The score you chose was: {score_choice}\n")

            print("nice turn. Your score is: ", self.calculate_score())

        else:
            self.sub_turn += 1
        print(self.__dict__)

        return

    def calculate_score(self):
        singles = ["ones", "twos", "threes", "fours", "fives", "sixes"]
        jokers = ["three_of_a_kind", "four_of_a_kind", "full_house", "small_straight", "large_straight", "yahtzee"]
        singles_sum = sum([self.__getattribute__(single) for single in singles])
        jokers_sum = sum([self.__getattribute__(joker) for joker in jokers])
        if singles_sum >= 63 and not self.get_bonus:
            self.get_bonus = True

        if self.get_bonus:
            self.total_score = singles_sum + jokers_sum + 63
        else:
            self.total_score = singles_sum + jokers_sum

        return self.total_score



    def temp_randomly_pick_score(self):
        """This method will pick one of the remaining score choices"""
        score_picks = ["ones", "twos", "threes", "fours", "fives", "sixes", "three_of_a_kind",
                       "four_of_a_kind", "full_house", "small_straight", "large_straight", "yahtzee"]
        pick = random.randint(0, 11)
        if self.__getattribute__(score_picks[pick]) == 0:
            return score_picks[pick]
        else:
            return self.temp_randomly_pick_score()

    def print_roll(self):
        print(self.first_roll, self.second_roll, self.third_roll)

    def print_scores(self):
        print("Nice game! Your overall score was: ", self.calculate_score())
        print("\nScorecard:\n")
        scorecard = ["ones", "twos", "threes", "fours", "fives", "sixes", "three_of_a_kind",
         "four_of_a_kind", "full_house", "small_straight", "large_straight", "yahtzee"]
        for item in scorecard:
            print(item, ": ", self.__getattribute__(item))

if __name__ == '__main__':
    test = Yahtzee()
    test.roll_dice()
    test.print_roll()

    start = time.perf_counter()
    for i in range(12):
        print("Turn: ", i)
        for y in range(3):
            test.turn(player_input=False)
    test.print_scores()
    print(f"Took {time.perf_counter() - start}s to play")
