import random

import numpy as np

# The testing script is full of boiler plate because I did not follow SOLID design principles when building the Yahtzee game
# This has a huge impact on unit testing as the Yahtzee class is a tangled mess - a monolithic thing.
# It also had an effect on development as it was more difficult to find and iron out bugs. Lesson learned!


class Yahtzee:
    """"
    Yahtzee Game Class. To use, call the class and pass in the type of player: choices are human, random or model
    Then for 12 turns, and 3 sub turns, call Yahtzee.turn()
    """
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
    chosen_scores = []

    def __init__(self, player_type: str = "random"):
        self.player_type: str = player_type  # Semantically encodes whether the player is a human, random or model
        # This in turn determines the paramterers fed to the method "turn"

        # TODO - none of these variables are actrually used
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

        self.roll_dice()

        return

    def roll_dice(self):
        numbers = [random.randint(1, 6) for i in range(15)]

        # commented to make super lightweight for training
        # if numbers.count(0) > 0:
        #     raise Exception("Dice roll contains 0. Random package screwed you buddy")

        # Reset the rolls
        self.first_roll = {"one": 0, "two": 0, "three": 0, "four": 0, "five": 0}
        self.second_roll = {"one": 0, "two": 0, "three": 0, "four": 0, "five": 0}
        self.third_roll = {"one": 0, "two": 0, "three": 0, "four": 0, "five": 0}

        for index, key in enumerate(self.first_roll):
            self.first_roll[key] = numbers[index]
        for index, key in enumerate(self.second_roll):
            self.second_roll[key] = numbers[index + 5]
        for index, key in enumerate(self.third_roll):
            self.third_roll[key] = numbers[index + 10]

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
            if dice_to_choose[key] and active_roll[key]:  # Abuse that 0.0 is falsey
                # Get the current roll
                self.dice_saved.append(active_roll[key])

                # eliminate the chosen dice from future rolls - very nested code is not great but not end of the world
                # if statement skips code block if its the last sub-roll on a turn (i.e. third roll of dice)
                if list_of_rolls_to_update is not None:
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
        number_of_chosen_for_unique_dice = (self.dice_saved.count(dice_value) for dice_value in unique_dice_chosen)
        if len(self.dice_saved) == 5 \
                and len(unique_dice_chosen) == 2 \
                and 3 in number_of_chosen_for_unique_dice:
            return 25
        return 0

    def pick_small_straight(self):
        """Pick a 4 in a row straight. abuse that there is only 3 possibilities for this, and that sets are sorted"""
        unique_dice_chosen = sorted(list(set(self.dice_saved)))
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
        """
        wrapper for picking scores. Currently works on string input.
        """
        if self.sub_turn != 3:
            raise Exception("Tried to pick a score but its not the third roll")
        # Python 3.10 - use switch statement
        match pick:
            case "ones":
                if "ones" in self.chosen_scores:
                    return None
                self.ones = self.pick_ones()
                self.chosen_scores.append("ones")
                return pick
            case "twos":
                if "twos" in self.chosen_scores:
                    return None
                self.twos = self.pick_twos()
                self.chosen_scores.append("twos")
                return pick
            case "threes":
                if "threes" in self.chosen_scores:
                    return None
                self.threes = self.pick_threes()
                self.chosen_scores.append("threes")
                return pick
            case "fours":
                if "fours" in self.chosen_scores:
                    return None
                self.fours = self.pick_fours()
                self.chosen_scores.append("fours")
                return pick
            case "fives":
                if "fives" in self.chosen_scores:
                    return None
                self.fives = self.pick_fives()
                self.chosen_scores.append("fives")
                return pick
            case "sixes":
                if "sixes" in self.chosen_scores:
                    return None
                self.sixes = self.pick_sixes()
                self.chosen_scores.append("sixes")
                return pick
            case "three_of_a_kind":
                if "three_of_a_kind" in self.chosen_scores:
                    return None
                self.three_of_a_kind = self.pick_three_of_a_kind()
                self.chosen_scores.append("three_of_a_kind")
                return pick
            case "four_of_a_kind":
                if "four_of_a_kind" in self.chosen_scores:
                    return None
                self.four_of_a_kind = self.pick_four_of_a_kind()
                self.chosen_scores.append("four_of_a_kind")
                return pick
            case "full_house":
                if "full_house" in self.chosen_scores:
                    return None
                self.full_house = self.pick_full_house()
                self.chosen_scores.append("full_house")
                return pick
            case "small_straight":
                if "small_straight" in self.chosen_scores:
                    return None
                self.small_straight = self.pick_small_straight()
                self.chosen_scores.append("small_straight")
                return pick
            case "large_straight":
                if "large_straight" in self.chosen_scores:
                    return None
                self.large_straight = self.pick_large_straight()
                self.chosen_scores.append("large_straight")
                return pick
            case "chance":
                if "chance" in self.chosen_scores:
                    return None
                self.chance = self.pick_chance()
                self.chosen_scores.append("chance")
                return pick
            case "yahtzee":
                if self.chosen_scores.count("yahtzee") > 1:  # Yahztee and bonus has already bene filled - move one
                    return None
                if "yahtzee" in self.chosen_scores:  # If yahtzee has been tried once add yahtzee bonus if it applies
                    self.check_yahtzee_bonus()
                else:  # Third case, no yahtzee has been tried, then case is like normal
                    self.chosen_scores.append("yahtzee")
                    self.yahtzee = self.pick_yahtzee()
                return pick
        return pick

    def turn(self, player_input=False, random_choice=False, choice_dice=None, choice_score=None, verbose: bool = False):
        """A single turn of the game"""
        if self.sub_turn == 1 and self.turn_number != 1:
            # Remove the first roll of the dice, do that when resetting the game
            self.dice_saved = []
            self.roll_dice()
        
        # When a player is choosing the move
        if player_input:
            print("\nThis is the current dice roll. 0's mean they cannot be selected")
            print("Dice roll: \n")
            self.print_roll()
            print("\n")
            print("You will now be asked for each dice whether to keep it or not. 1 is keep.")
            die = ["one", "two", "three", "four", "five"]
            choice = {}
            for index, dice in enumerate(die):
                choice[dice] = int(input(f"Input a 0 or 1 for dice number {dice}"))
        elif not random_choice:
            # Choices are input as an argument by the QNNetwork
            choice = choice_dice
        else:
            # Bodge: get the current roll
            rolls = {1: "first_roll", 2: "second_roll", 3: "third_roll"}
            current_roll = self.__getattribute__(rolls[self.sub_turn])

            # Randomly select the dice that remain
            if self.sub_turn != 3:
                choice = {key: (1 if (random.random() > 0.5 and val != 0) else 0) for key, val in current_roll.items()}
            else:
                # If last turn just pick all the remaining dice
                choice = {key: 1 if val != 0 else 0 for key, val in current_roll.items()}

        if verbose or player_input:  # Utility for printing results to the terminal
            print("Your choice was: ", choice)
        self.pick_dice(choice)

        # If its the third dice roll i.e. sub turn 3, then pick a score!
        if self.sub_turn == 3:
            if player_input:
                print("Your chosen dice are: \n")
                print(self.dice_saved)
                score_choice = input("""input a score choice. The available are: singles (e.g. ones), three_of_a_kind, 
                four_of_a_kind, full_house, small_straight, large_straight, and yahtzee.""")
            elif not random_choice:
                # Choices are input as an argument by the QNNetwork
                score_choice = choice_score
            else:
                # Random player
                scores_to_choose = ["ones", "twos", "threes", "fours", "fives", "sixes", "three_of_a_kind",
                                    "four_of_a_kind", "full_house", "small_straight", "large_straight", "yahtzee",
                                    "chance"]
                random_choice = [random.random() for i in range(len(scores_to_choose))]
                score_choice = scores_to_choose[np.argmax(random_choice)]

            score = self.pick_score(score_choice)
            self.turn_number += 1
            self.sub_turn = 1
            self.dice_saved = []

            if verbose or player_input:
                print(f"The score you chose was: {score_choice}\n")
                print("nice turn. Your score is: ", self.calculate_score())
        else:
            score = 0
            self.sub_turn += 1

        return score

    def calculate_score(self):
        singles = ["ones", "twos", "threes", "fours", "fives", "sixes"]
        jokers = ["three_of_a_kind", "four_of_a_kind", "full_house", "small_straight", "large_straight", "yahtzee","chance"]
        singles_sum = sum([self.__getattribute__(single) for single in singles])
        jokers_sum = sum([self.__getattribute__(joker) for joker in jokers])
        if singles_sum >= 63 and not self.get_bonus:
            self.get_bonus = True
            self.total_score = singles_sum + jokers_sum + 63
        else:
            self.total_score = singles_sum + jokers_sum

        return self.total_score

    def print_roll(self):
        print(self.first_roll, self.second_roll, self.third_roll)

    def print_scores(self, verbose=True) -> dict:
        scorecard = ["ones", "twos", "threes", "fours", "fives", "sixes", "three_of_a_kind",  # TODO Enum this
                     "four_of_a_kind", "full_house", "small_straight", "large_straight", "yahtzee", "chance"]
        if not verbose:
            return {item: self.__getattribute__(item) for item in scorecard}
        print("Nice game! Your overall score was: ", self.calculate_score())
        print("\nScorecard:\n")

        for item in scorecard:
            print(item, ": ", self.__getattribute__(item))
        
        return

    def reset_game(self):
        self.turn_number = 1
        self.sub_turn = 1
        self.dice_saved = []
        self.chosen_scores = []  # 9 February added this tracker of chosen scores

        self.third_roll, self.second_roll, self.first_roll = self.empty_roll.copy(), self.empty_roll.copy(), self.empty_roll.copy()
        self.roll_dice()

        score_card = ["ones", "twos", "threes", "fours", "fives", "sixes", "three_of_a_kind", "four_of_a_kind",
                      "full_house", "small_straight", "large_straight", "yahtzee", "chance", "yahtze_bonus",
                      "singles_total", "total_score"]
        for item in score_card:
            self.__setattr__(item, 0)
        self.get_bonus = False


def random_player_game(random_player: Yahtzee):
    random_player.roll_dice()
    for i in range(12):
        for y in range(3):
            if random_player.sub_turn == 1:
                random_player.roll_dice()
            random_player.turn(player_input=False, random_choice=True)
    score = random_player.calculate_score()
    random_player.reset_game()
    return score


if __name__ == "__main__":
    game = Yahtzee(player_type="Human")

    # for i in range(13):
    #     game.turn(player_input=True)

    random_player = Yahtzee(player_type="random")
    random_results = [random_player_game(random_player=random_player) for i in range(2)]
