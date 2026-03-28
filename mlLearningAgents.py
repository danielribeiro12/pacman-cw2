# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        self.pacmanPos = state.getPacmanPosition()
        self.ghostPos = tuple(sorted(state.getGhostPositions()))

        food = state.getFood()
        self.food = tuple(tuple(row) for row in food)

        # store legal actions here so maxQValue doesn't need the full GameState
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        self.legalActions = legal

    def __hash__(self):
        return hash((self.pacmanPos, self.ghostPos, self.food))

    def __eq__(self, other):
        if not isinstance(other, GameStateFeatures):
            return False
        return (self.pacmanPos == other.pacmanPos and
                self.ghostPos == other.ghostPos and
                self.food == other.food)


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # Q-table: (state, action) -> float, defaults to 0.0 for unseen pairs
        self.qValues = {}

        # visit counts per (state, action), used by explorationFn
        self.counts = {}

        # store last state/action so we can compute the reward on the next step
        self.prevState = None
        self.prevAction = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        return endState.getScore() - startState.getScore()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        return self.qValues.get((state, action), 0.0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        legal = state.legalActions
        if not legal:
            return 0.0
        return max(self.getQValue(state, a) for a in legal)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        current_q = self.getQValue(state, action)

        # terminal state has no future value
        max_next_q = 0.0 if nextState is None else self.maxQValue(nextState)

        # TD target: r + gamma * max Q(s')
        td_target = reward + self.gamma * max_next_q

        # nudge Q(s,a) towards the TD target by alpha
        new_q = current_q + self.alpha * (td_target - current_q)
        self.qValues[(state, action)] = new_q

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        key = (state, action)
        self.counts[key] = self.counts.get(key, 0) + 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Returns the number of times (state, action) has been visited.
        Returns 0 for unseen pairs.

        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.counts.get((state, action), 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        if counts < self.maxAttempts:
            # least-pick: always prefer the least-visited action until it has been
            # tried maxAttempts times, then fall back to the Q-value
            return float('inf')
        return utility

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # remove STOP, never want pacman to stand still
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        stateFeatures = GameStateFeatures(state)

        # run the Q-update for the previous (s, a, r, s') transition
        if self.prevState is not None:
            reward = self.computeReward(self.prevState, state)
            prevFeatures = GameStateFeatures(self.prevState)
            self.learn(prevFeatures, self.prevAction, reward, stateFeatures)

        # epsilon-greedy: explore randomly with prob epsilon,
        # otherwise pick the action with the best explorationFn score
        if util.flipCoin(self.epsilon):
            action = random.choice(legal)
        else:
            if self.alpha > 0:
                # training: explorationFn returns inf for under-visited actions
                # so they're always tried first; falls back to Q-value once explored
                scores = [self.explorationFn(self.getQValue(stateFeatures, a),
                                             self.getCount(stateFeatures, a))
                          for a in legal]
            else:
                # testing: ignore exploration bonus, act purely on learned Q-values
                scores = [self.getQValue(stateFeatures, a) for a in legal]
            best_score = max(scores)
            best_actions = [a for a, s in zip(legal, scores) if s == best_score]
            action = random.choice(best_actions)

        # update count and save for next step's Q-update
        self.updateCount(stateFeatures, action)
        self.prevState = state
        self.prevAction = action

        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # final Q-update for the terminal transition — no next state so max Q(s') = 0
        if self.prevState is not None:
            reward = self.computeReward(self.prevState, state)
            prevFeatures = GameStateFeatures(self.prevState)
            self.learn(prevFeatures, self.prevAction, reward, None)

        # reset for next episode
        self.prevState = None
        self.prevAction = None

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
