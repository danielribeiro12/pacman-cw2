# Q-Learning implementation for Pacman game

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    State features for Q-learning - extracts relevant information from game state
    """

    def __init__(self, state: GameState):
        """Initialize features from game state"""
        self.pacmanPos = state.getPacmanPosition()
        self.ghostPos = tuple(state.getGhostPositions())
        
        # Extract food positions
        foodGrid = state.getFood()
        foods = []
        for x in range(foodGrid.width):
            for y in range(foodGrid.height):
                if foodGrid[x][y]:
                    foods.append((x, y))
        self.foodPos = tuple(foods)
    
    def __hash__(self):
        return hash((self.pacmanPos, self.ghostPos, self.foodPos))
    
    def __eq__(self, other):
        if not isinstance(other, GameStateFeatures):
            return False
        return (self.pacmanPos == other.pacmanPos and 
                self.ghostPos == other.ghostPos and 
                self.foodPos == other.foodPos)


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        Initialize Q-learning agent with parameters for learning
        alpha: learning rate
        epsilon: exploration probability
        gamma: discount factor
        numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0
        
        # Q-values: dictionary mapping (state, action) to value
        self.qValues = {}
        # Count visits to each state-action pair
        self.counts = {}
        
        # Track previous state for learning
        self.prevState = None
        self.prevAction = None
        self.prevStateFeatures = None

    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

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

    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        """
        Calculate reward signal for state transition
        Returns: reward value
        """
        # Check terminal states
        if endState.isWin():
            return 500.0
        if endState.isLose():
            return -500.0
        
        reward = 0.0
        
        # Check if food was eaten
        if endState.getNumFood() < startState.getNumFood():
            reward += 10.0
        
        # Penalty for taking a step
        reward -= 1.0
        
        return reward

    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        """Get Q-value for state-action pair"""
        return self.qValues.get((state, action), 0.0)

    def maxQValue(self, state: GameStateFeatures) -> float:
        """Get maximum Q-value for a state"""
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        values = []
        for action in actions:
            values.append(self.getQValue(state, action))
        
        if len(values) > 0:
            return max(values)
        return 0.0

    def learn(self, state: GameStateFeatures, action: Directions, 
              reward: float, nextState: GameStateFeatures):
        """
        Update Q-values using Q-learning rule
        Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_Q(s') - Q(s,a))
        """
        # Get current Q-value
        currentQ = self.getQValue(state, action)
        
        # Get max Q-value for next state
        maxNextQ = self.maxQValue(nextState)
        
        # Calculate new Q-value
        newQ = currentQ + self.alpha * (reward + self.gamma * maxNextQ - currentQ)
        
        # Store the new Q-value
        self.qValues[(state, action)] = newQ

    def updateCount(self, state: GameStateFeatures, action: Directions):
        """Increment count for state-action pair"""
        key = (state, action)
        if key not in self.counts:
            self.counts[key] = 0
        self.counts[key] += 1

    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        """Get count for state-action pair"""
        return self.counts.get((state, action), 0)

    def explorationFn(self, utility: float, counts: int) -> float:
        """
        Compute exploration value to balance exploitation and exploration
        Higher for actions that haven't been tried much
        """
        return utility + 1.0 / (counts + 1.0)

    def getAction(self, state: GameState) -> Directions:
        """
        Choose action using epsilon-greedy with exploration function
        """
        # Get legal actions
        legalActions = state.getLegalPacmanActions()
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)
        
        # Get state features
        stateFeatures = GameStateFeatures(state)
        
        # Learn from previous transition if not first move
        if self.prevState is not None:
            reward = self.computeReward(self.prevState, state)
            self.learn(self.prevStateFeatures, self.prevAction, reward, stateFeatures)
        
        # Choose action with epsilon-greedy
        if util.flipCoin(self.epsilon):
            # Random action for exploration
            action = random.choice(legalActions)
        else:
            # Pick best action according to exploration function
            bestValue = float('-inf')
            bestAction = None
            
            for a in legalActions:
                q = self.getQValue(stateFeatures, a)
                c = self.getCount(stateFeatures, a)
                exploreValue = self.explorationFn(q, c)
                
                if exploreValue > bestValue:
                    bestValue = exploreValue
                    bestAction = a
            
            if bestAction is None:
                action = random.choice(legalActions)
            else:
                action = bestAction
        
        # Update count and store for next iteration
        self.updateCount(stateFeatures, action)
        self.prevState = state
        self.prevStateFeatures = stateFeatures
        self.prevAction = action
        
        return action

    def final(self, state: GameState):
        """
        Called at end of game to finalize learning
        """
        # Learn final transition
        if self.prevState is not None:
            reward = self.computeReward(self.prevState, state)
            finalStateFeatures = GameStateFeatures(state)
            self.learn(self.prevStateFeatures, self.prevAction, reward, finalStateFeatures)
        
        # Reset for next episode
        self.prevState = None
        self.prevStateFeatures = None
        self.prevAction = None
        
        # Update episode counter and stop learning when done
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            self.setAlpha(0)
            self.setEpsilon(0)