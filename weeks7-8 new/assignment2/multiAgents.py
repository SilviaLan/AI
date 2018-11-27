# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    # print Agent

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states

        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        "*** YOUR CODE HERE ***"
        distance = []
        # foodList = currentGameState.getFood()
        foodList = currentGameState.getFood().asList()
        # print "!!!!!!!!"
        # print foodList
        # print "!!!!!!"
        pacmanPos = list(successorGameState.getPacmanPosition()) #newPos,
        # print "11111"
        # print pacmanPos
        # print "11111"
        # The list () method is used to convert tuples into lists.

        if action == 'Stop':
            return -float("inf")

        for ghostState in newGhostStates:
            if ghostState.getPosition() == tuple(pacmanPos) and ghostState.scaredTimer is 0:
                return -float("inf")

        for food in foodList:
            x = -1 * abs(food[0] - pacmanPos[0])
            y = -1 * abs(food[1] - pacmanPos[1])
            distance.append(x + y)

        return max(distance)

        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        v = float('-inf')
        nextAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            # print "1111111"
            # # print action
            # print gameState.getLegalActions(0)
            # print "1111111"
            temp = self.minValue(0, 1, gameState.generateSuccessor(0, action))
            # print temp
            if temp > v and action != Directions.STOP:
                v = temp
                nextAction = action

        return nextAction

    def maxValue(self, depth, agent, state):
        if depth == self.depth:
            return self.evaluationFunction(state)
        else:
            actions = state.getLegalActions(agent)
            if len(actions) > 0:
                v = float('-inf')
            else:
                v = self.evaluationFunction(state)
            for action in actions:
                s = self.minValue(depth, agent + 1, state.generateSuccessor(agent, action))
                if s > v:
                    v = s
            return v

    def minValue(self, depth, agent, state):
        # print "!!!!"
        # print agent
        # print "!!!!"
        if depth == self.depth:
            return self.evaluationFunction(state)
        else:
            actions = state.getLegalActions(agent)
            if len(actions) > 0:
                v = float('inf')
            else:
                v = self.evaluationFunction(state)

            for action in actions:
                if agent == state.getNumAgents() - 1:
                    s = self.maxValue(depth + 1, 0, state.generateSuccessor(agent, action))
                    if s < v:
                        v = s
                else:
                    s = self.minValue(depth, agent + 1, state.generateSuccessor(agent, action))
                    if s < v:
                        v = s
            return v

        # version 2
        # def value(state, agentIndex, depth):
        #     if depth == 0 or state.isWin() or state.isLose():
        #         return [self.evaluationFunction(state), None]
        #
        #     terminalState = state.getNumAgents() - 1
        #     nextAgent = 0 if agentIndex == terminalState else agentIndex + 1
        #     d = depth - 1 if agentIndex == terminalState else depth
        #
        #     actions = state.getLegalActions(agentIndex)
        #     nextValue = []
        #     for a in actions:
        #         successor = value(state.generateSuccessor(agentIndex, a), nextAgent, d)[0]
        #         nextValue.append(successor)
        #
        #     if agentIndex == 0:  # max state
        #         return [max(nextValue), actions[nextValue.index(max(nextValue))]]
        #     else:  # min state
        #         return [min(nextValue), actions[nextValue.index(min(nextValue))]]
        #
        # bestAction = value(gameState, self.index, self.depth)[1]
        # return bestAction
             
        # util.raiseNotDefined()



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      An expectimax agent you can use or modify
    """
    def expectimax_value (self, gameState, agentIndex, nodeDepth):
        if(agentIndex >= gameState.getNumAgents()):
            agentIndex = 0
            nodeDepth += 1
        if( nodeDepth == self.depth ):
            return self.evaluationFunction(gameState)
        if( agentIndex == self.index ):
            return self.max_value(gameState, agentIndex, nodeDepth)

        else:
            return self.exp_value(gameState, agentIndex, nodeDepth)

        return 'None'

    def max_value (self, gameState, agentIndex, nodeDepth):
      if( gameState.isWin() or gameState.isLose() ):
        return self.evaluationFunction(gameState)

      value = float("-inf")
      actionValue = "Stop"

      for legalActions in gameState.getLegalActions(agentIndex) :
        if legalActions == Directions.STOP:
          continue
        successor = gameState.generateSuccessor(agentIndex, legalActions)
        temp = self.expectimax_value(successor, agentIndex+1, nodeDepth)
        if( temp > value):
          value = temp
          actionValue = legalActions

      if( nodeDepth == 0 ):
        return actionValue
      else:
        return value

    def exp_value(self, gameState, agentIndex, nodeDepth):
      if( gameState.isWin() or gameState.isLose() ):
        return self.evaluationFunction(gameState)
      value = 0
      pr = 1.0/len(gameState.getLegalActions(agentIndex))

      for legalActions in gameState.getLegalActions(agentIndex):
        if( legalActions == Directions.STOP):
          continue
        successor = gameState.generateSuccessor(agentIndex, legalActions)
        value = value + (self.expectimax_value(successor, agentIndex+1, nodeDepth) * pr)
      return value

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.expectimax_value(gameState,0,0)
    

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 3). 

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = currentGameState.getScore()
    eatenGhost = 200
    ghostScore = 0
    ghostWeight = 10
    foodWeight = 10

    for ghost in newGhostStates:
        ghostDistance = manhattanDistance(newPos, newGhostStates[0].getPosition())
        # print "!!!!"
        # print ghostDistance

        if ghostDistance:

            if ghost.scaredTimer:
                update = eatenGhost / float(ghostDistance)
                ghostScore = ghostScore + update
            else:
                update = ghostWeight / ghostDistance
                ghostScore = ghostScore - update


    score = score + ghostScore
    foodDistance = [manhattanDistance(newPos, food) for food in newFood.asList()]

    if foodDistance:
        update = foodWeight / float(min(foodDistance))
        score = score + update

    return score

    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

