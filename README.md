# Table of contents
1. [Overview](#Yahtzee_AI)
2. [Inspiration and goals](#Inspiration)
3. [Model architecture and approach](#Model-architecture)
4. [Development and experimentation](#Development-and-experimentation)
5. [Installation and usage](#Installation-and-usage)
6. [References](#References)
7. [Disclaimer](#Disclaimer-and-license)

![Final score](https://github.com/byrnesy924/Yatzhee_AI/assets/89000131/93157c57-05bc-43ef-a47b-5b0957a3d9a4)
![Special scores over time](https://github.com/byrnesy924/Yatzhee_AI/assets/89000131/bb1f376d-674f-47e2-922f-0b65565f2516)

TODO check spelling in word

# Yahtzee_AI: Overview of the Double Deep Q Learning Agent
This project creates and trains a deep learning agent to play the game of Yatzhee. Yahtzee is primarily a game of chance, but also gives rise to complex tactics and strategies through player choice. Although the game is simple, the progressive nature of allocating dice rolls into a specific score choice means that the state space of the game is significant and increases exponentially with the number of players - there are 19 billion unique states in a single player game and $≈ (213·100)^n · 6^5· 3$ unique states in an n player game (Kang, Schroeder 2018) #TODO CHECK. It requires players to make both strategic and tactical decisions and balance their long term and short term interests.

After beginning development of this project and during research for it, I came across the following paper that takes a similar approach. Some other approaches and resources are listed below # TODO check language here.
[2018 stanford paper](https://web.stanford.edu/class/aa228/reports/2018/final75.pdf)
[Yale publication and the work of James Glenn] https://raw.githubusercontent.com/philvasseur/Yahtzee-DQN-Thesis/dcf2bfe15c3b8c0ff3256f02dd3c0aabdbcbc9bb/webpage/final_report.pdf

<a id="Inspiration"></a>
# Inspiration and goals

The inspiration for this project came after playing Yahtzee with my partner's family, and was my first real experiments into deep learning and reinforcement learning. 

The goal of this project was to upskill in deep learning, specifically in TensorFlow and Keras, and along to way learn as much as possible about machine learning in production, reinforcement learning, Q learning, hyperparameter tuning of a machine learning model, gaussian processes and Bayesian optimisation.

Inspiration, Inception of idea, presented initial results to work colleagues in a presentation (maybe link presentation? might need to remove branding)

<a id="Model-architecture"></a>
# Model architecture and approach
The approach is a Double Deep Q Learning Method. To break this down:
- Q learning is a type of reinforcement learning. It is a model-free approach (rephrased, the agent does not assume anything in way of a model - it instead must learn soley from the environment)
- Deep refers to the use of a neural network to impliment the Q learning algorithm.
    - Using a neural network is not necessary, but as per (TODO cite) below is necessary
- Double refers to using two networks - a normal model and a target
    - This is a technique that reduces maximisation bias and can improve policy choice
 
In a bit more detail:
The agent functions as a matrix transformation from the state space (the mathematical representation of all the different states of the games and the possible choices at each state), to the _reward space_  
The reward space is defined by the reward function, which takes in an action (decided by the agent) and provides a reward
The point of the agent is to maximise the reward.
This means that in reinforcement learning, alignment (specifically in this case outer alignment) is very important. This is because the reward function is just a proxy for what we actually want to agent to be capable of - **being a good yahtzee player** 

Following this, the approach taken is a heirarchical approach (Max Q learning). 
In this, the agent is rewarded for doing things like choosing scores and
    choosing dice, but also rewarded for the score at the end of each turn and their overall score in the game
    i.e. Q(s, a) = V(s, a) + C(s,a) where V is a subtask (e.g. choosing dice) and C is a completion task

This:
- reduces the sparcity of rewards
- improves performance, learning and the final policy choice
- **improves alignemnt by improving the ability of the agent, rather than just giving it only the raw reward of its Yahtzee score** (as long as we carefully choose the reward function to improve policy)

Some resources for Double Deep Q learning:
    - https://www.semanticscholar.org/paper/Deep-Reinforcement-Learning-with-Double-Q-Learning-Hasselt-Guez/3b9732bb07dc99bde5e1f9f75251c6ea5039373e
    - https://arxiv.org/abs/1509.06461
    - https://dl.acm.org/doi/10.5555/3016100.3016191
    - https://ai.stackexchange.com/questions/21515/is-there-any-good-reference-for-double-deep-q-learning

<a id="Development-and-experimentation"></a>
# Development and experimentation
The first step of development was creating a simple implimentation of the Yahtzee game. A few notes about approach:
- The dice are pre-rolled before each turn, using python's random module
- The game is divided into 13 turns, with 3 sub turns. Each sub turn represents each oppertunity to roll the dice. Naturally, if you have chosen all your dice in a turn then you cannot choose again. However, it was easier to impliment every sub-turn, and structure the reward function is such a way so that the agent learns how to approch each sub turn.
    - I believe in one of the resources the approach is to use two different models - one to choose when to re-roll the dice, and another to choose which dice

TODO
- Creating the NNQ Model
    - Mathematical explanation

Once the model agent was built and functional, I went down the path of tuning the hyperparameters. This was my first time doing it, and tried a researched a few different approaches:
- Grid searching the hyperparameter space
- Randomly searching the hyperparameter space
- Using Bayesian optimisation to search the hyperparameter space

In this repository is a random search method, and a bayesian optimisation method.
The optimisation method produced a number of results, but also had some limitations given the package used and the amount of compute I had access to:
- Higher learning rates and lower gamma significantly increased performance over a short training period (less than 16 epochs - 1024 games)
- Hyperparameter testing helped me narrow in on the heirarchical structure of the reward function
    - See also curse of dimensionality below    
- There was a signifcant amount of noise involved with training, partially due to random initialisation of the nueral network
    - This meant Bayesian Optimisation was a much better approach than random searching
- Because of the high number of hyperparameters (reward factors, model hyperparameters like learning rate, gamma, length of memory, batch size, model architecture etc) improving the performance of hyperparameters was difficult with limited access to compute
- Another concern of mine was potential non-linearity of learning; performance in the first 1000 games does not necessarily translate to optimal policy choice
See below for a visualisation of the hyper-parameter tuning
# TODO - Visualisation
# TODO Grokking and long training
- Experiments with grokking - training for longer with stronger hardware

# Installation and usage
TBD
- Playing the game
    - install and so on

# References
Lorem ipsum
[2018 stanford paper](https://web.stanford.edu/class/aa228/reports/2018/final75.pdf)
[Yale publication and the work of James Glenn and PhD student Phil Vasseur](https://raw.githubusercontent.com/philvasseur/Yahtzee-DQN-Thesis/dcf2bfe15c3b8c0ff3256f02dd3c0aabdbcbc9bb/webpage/final_report.pdf)
[Initially used to help write code](https://medium.com/@carsten.friedrich/part-4-neural-network-q-learning-a-tic-tac-toe-player-that-learns-kind-of-2090ca4798d)
[Great article on using a double q learning model](https://medium.com/p/b6bf911b6b2c)
[Another Q Learning approach related to Yahtzee](https://www.yahtzeemanifesto.com/reinforcement-learning-yahtzee.pdf)
    ^ Where methods differ is in the implimentation of the game and heirarchy. This project assumes reroll always,
    which is more computationally expensive, but the agent does not need to learn to choose to reroll
    so it is easier to impliment

# Disclaimer and license
Lorem ipsum

