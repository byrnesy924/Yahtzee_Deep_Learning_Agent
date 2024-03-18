# Table of contents
1. [Overview](#Yahtzee_AI)
2. [Inspiration and goals](#Inspiration)
2. [Installation and usage](#Installation-and-usage)
3. [Model architecture](#Model-architecture)
4. [Development and experimentation](#Development-and-experimentation)
5. [References](#References)
6. [Disclaimer](#Disclaimer-and-license)

# Yahtzee_AI: Overview of the Double Deep Q Learning Agent
This project creates and trains a deep learning agent to play the game of Yatzhee. Yahtzee is involves a significant amount of probability, while also giving rise to complex tactics and strategies through player choice. Although the game is simple, the progressive nature of allocating dice rolls into a specific score choice means that the state space of the game is large and increases exponentially with the number of players - there are 19 billion unique states in a single player game and $≈ (213·100)^n · 6^5· 3$ unique states in an n player game (Kang, Schroeder 2018) #TODO CHECK. It requires players to make both strategic and tactical decisions and balance their long term and short term interests.

After beginning development of this project and during research for it, I came across the following paper that takes a similar approach # TODO check language here.
[2018 stanford paper](https://web.stanford.edu/class/aa228/reports/2018/final75.pdf)
[Yale publication and the work of James Glenn] https://raw.githubusercontent.com/philvasseur/Yahtzee-DQN-Thesis/dcf2bfe15c3b8c0ff3256f02dd3c0aabdbcbc9bb/webpage/final_report.pdf

<a id="Inspiration"></a>
# Inspiration and goals

The inspiration for this project came after playing Yahtzee with my partner's family, and was my first real experiments into deep learning and reinforcement learning. 

The goal of this project was to upskill in deep learning, specifically in TensorFlow and Keras, and along to way learn as much as possible about machine learning in production, reinforcement learning, Q learning, hyperparameter tuning of a machine learning model, gaussian processes and Bayesian optimisation.

Inspiration, Inception of idea, presented initial results to work colleagues in a presentation (maybe link presentation? might need to remove branding)

![Final score](https://github.com/byrnesy924/Yatzhee_AI/assets/89000131/93157c57-05bc-43ef-a47b-5b0957a3d9a4)
![Special scores over time](https://github.com/byrnesy924/Yatzhee_AI/assets/89000131/bb1f376d-674f-47e2-922f-0b65565f2516)

# Installation and usage
TBD
- Playing the game
    - install and so on

# Model architecture
Lorem ipsum

# Development and experimentation
Lorem ipsum
- Creating the Yahtzee Game
- Creating the NNQ Model
    - Mathematical explanation
- Initial training and testing
    - findings about learning rate, plus other observations about double Q Learning and all that
- Hyperparameter testing
    - Initial random approach
    - Bayesian Approach - noisey black box function
    - preliminary findings
    - curse of dimensionality - implimenting the moving windows of boundaries; exploration vs exploitation
- Experiments with grokking - training for longer with stronger hardware

# References
Lorem ipsum
[2018 stanford paper](https://web.stanford.edu/class/aa228/reports/2018/final75.pdf)
[Yale publication and the work of James Glenn] https://raw.githubusercontent.com/philvasseur/Yahtzee-DQN-Thesis/dcf2bfe15c3b8c0ff3256f02dd3c0aabdbcbc9bb/webpage/final_report.pdf
https://raw.githubusercontent.com/philvasseur/Yahtzee-DQN-Thesis/dcf2bfe15c3b8c0ff3256f02dd3c0aabdbcbc9bb/webpage/final_report.pdf


# Disclaimer and license
Lorem ipsum

