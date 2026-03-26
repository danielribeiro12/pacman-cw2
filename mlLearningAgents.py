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
        Extracts the relevant features from a game state and stores them in
        a hashable form so that GameStateFeatures can be used as a dict key.

        Features captured:
          - Pacman's position
          - All ghost positions (as a sorted tuple for canonical ordering)
          - The food grid (as a tuple of tuples of bools)
          - The set of legal non-STOP actions (stored for maxQValue use)

        Args:
            state: A given game state object
        """
        self.pacmanPos = state.getPacmanPosition()

        # Sort ghost positions so state is canonical regardless of ghost ordering
        self.ghostPos = tuple(sorted(state.getGhostPositions()))

        # Convert the food grid to a hashable nested tuple
        food = state.getFood()
        self.food = tuple(tuple(row) for row in food)

        # Store legal actions (excluding STOP) so maxQValue can enumerate them
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

    def __repr__(self):
        return f"GSF(pac={self.pacmanPos}, ghosts={self.ghostPos})"


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

        # Q-value table: maps (GameStateFeatures, action) -> float
        # Unseen pairs default to 0.0
        self.qValues = {}

        # Visitation counts: maps (GameStateFeatures, action) -> int
        # Used by explorationFn to encourage trying under-visited actions
        self.counts = {}

        # Previous-step bookkeeping for computing rewards and performing updates
        self.prevState = None   # GameState from the previous step
        self.prevAction = None  # action taken at the previous step

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
        The reward for a transition is simply the change in game score.
        The game score increases when Pacman eats food/ghosts and decreases
        each time step (to encourage efficiency) or on death.

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
        Returns the Q-value for (state, action). Unseen pairs return 0.0,
        which acts as an optimistic initialisation when rewards are negative.

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
        Returns the maximum Q-value over all legal (non-STOP) actions in the
        given state. Returns 0.0 for terminal states (no legal actions).

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
        Performs a one-step Q-learning update:

            Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))

        When nextState is None (terminal transition) max_a' Q(s', a') is treated as 0.

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state (None if terminal)
            reward: the reward received on this trajectory
        """
        current_q = self.getQValue(state, action)

        # For terminal transitions there is no future reward
        max_next_q = 0.0 if nextState is None else self.maxQValue(nextState)

        # TD target: immediate reward + discounted future value
        td_target = reward + self.gamma * max_next_q

        # Update rule: move Q-value towards the TD target
        new_q = current_q + self.alpha * (td_target - current_q)
        self.qValues[(state, action)] = new_q

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Increments the visitation count for (state, action).

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
        Optimistic exploration bonus: if a (state, action) pair has been
        visited fewer than maxAttempts times we return +infinity so that
        action is always preferred over already-explored alternatives.
        Once it has been tried enough, we fall back to the plain Q-value.

        This is a "least-pick" strategy: always try under-explored actions first.

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        if counts < self.maxAttempts:
            # Not yet tried enough: force exploration of this action
            return float('inf')
        return utility

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Selects an action combining epsilon-greedy exploration with the
        exploration function:
          - With probability epsilon: pick a uniformly random legal action.
          - With probability 1 - epsilon: pick the action that maximises
            explorationFn(Q(s, a), count(s, a)), breaking ties randomly.
            explorationFn prioritises under-visited actions (count < maxAttempts)
            over purely greedy Q-value selection, ensuring every action is tried
            a sufficient number of times before the agent commits to a policy.

        Before choosing, performs a Q-learning update using the previous
        (state, action, reward) transition if one exists.

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # Legal actions for Pacman, excluding STOP
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        stateFeatures = GameStateFeatures(state)

        # Q-learning update: if we have a previous transition, compute the reward
        # (score difference) and update Q(prevState, prevAction)
        if self.prevState is not None:
            reward = self.computeReward(self.prevState, state)
            prevFeatures = GameStateFeatures(self.prevState)
            self.learn(prevFeatures, self.prevAction, reward, stateFeatures)

        # Epsilon-greedy action selection
        # During training epsilon > 0 ensures some random exploration.
        # During testing epsilon = 0 so we always act via explorationFn.
        if util.flipCoin(self.epsilon):
            # Exploration: random action
            action = random.choice(legal)
        else:
            # Use explorationFn to score each action: under-visited actions
            # receive +inf, fully-visited actions are scored by their Q-value.
            # Pick the action with the highest exploration score; break ties randomly.
            scores = [self.explorationFn(self.getQValue(stateFeatures, a),
                                         self.getCount(stateFeatures, a))
                      for a in legal]
            best_score = max(scores)
            best_actions = [a for a, s in zip(legal, scores) if s == best_score]
            action = random.choice(best_actions)

        # Record visit and store transition info for the next step
        self.updateCount(stateFeatures, action)
        self.prevState = state
        self.prevAction = action

        return action

    def final(self, state: GameState):
        """
        Called by the game engine at the end of every episode (win or loss).

        Performs the final Q-learning update for the terminal transition
        (where the next-state value is 0 by definition), then resets
        the episode bookkeeping.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Terminal Q-learning update: no future reward from a terminal state
        if self.prevState is not None:
            reward = self.computeReward(self.prevState, state)
            prevFeatures = GameStateFeatures(self.prevState)
            # nextState=None signals that max Q(s', .) = 0
            self.learn(prevFeatures, self.prevAction, reward, None)

        # Reset episode bookkeeping ready for the next episode
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
