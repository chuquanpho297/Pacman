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
import numpy as np
from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
          def _scoreFromGhost(gameState):
            score = 0
            for ghost in gameState.getGhostStates():
              disGhost = manhattanDistance(gameState.getPacmanPosition(), ghost.getPosition())
              if ghost.scaredTimer > 0:
                score += pow(max(8 - disGhost, 0), 2)
              else:
                score -= pow(max(7 - disGhost, 0), 2)
            return score

          def _scoreFromFood(gameState):
            disFood = []
            for food in gameState.getFood().asList():
              disFood.append(1.0/manhattanDistance(gameState.getPacmanPosition(), food))
            if len(disFood)>0:
              return max(disFood)
            else:
              return 0

          def _scoreFromCapsules(gameState):
            score = []
            for Cap in gameState.getCapsules():
              score.append(50.0/manhattanDistance(gameState.getPacmanPosition(), Cap))
            if len(score) > 0:
              return max(score)
            else:
              return 0

          def _suicide(gameState):
            score = 0
            disGhost = 1e6
            for ghost in gameState.getGhostStates():
              disGhost = min(manhattanDistance(gameState.getPacmanPosition(), ghost.getPosition()), disGhost)
            score -= pow(disGhost, 2)
            if gameState.isLose():
              score = 1e6
            return score
          successorGameState = currentGameState.generatePacmanSuccessor(action)
          score = successorGameState.getScore()
          scoreGhosts = _scoreFromGhost(successorGameState)
          scoreFood = _scoreFromFood(successorGameState)
          scoreCapsules = _scoreFromCapsules(successorGameState)
          if score < 800 and currentGameState.getNumFood() <= 1 and len(currentGameState.getCapsules()) == 0:
            return _suicide(currentGameState)
          else:
            return score + scoreGhosts + scoreFood + scoreCapsules    

def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        numAgent = gameState.getNumAgents()
        ActionScore = []
        def _miniMax(s, iterCount):
          if iterCount >= self.depth*numAgent or s.isWin() or s.isLose():
            return betterEvaluationFunction(s)
          if iterCount%numAgent != 0: #Ghost min
            result = 1e10
            for a in s.getLegalActions(iterCount%numAgent):
              sdot = s.generateSuccessor(iterCount%numAgent,a)
              result = min(result, _miniMax(sdot, iterCount+1))
            return result
          else: # Pacman Max
            result = -1e10
            for a in s.getLegalActions(iterCount%numAgent):
              sdot = s.generateSuccessor(iterCount%numAgent,a)
              result = max(result, _miniMax(sdot, iterCount+1))
              if iterCount == 0:
                ActionScore.append(result)
            return result
          
        result = _miniMax(gameState, 0);
        return gameState.getLegalActions(0)[ActionScore.index(max(ActionScore))]

class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        numAgent = gameState.getNumAgents()
        ActionScore = []

        def _alphaBeta(s, iterCount, alpha, beta):
          if iterCount >= self.depth*numAgent or s.isWin() or s.isLose():
            return betterEvaluationFunction(s)
          if iterCount%numAgent != 0: #Ghost min
            result = 1e10
            for a in s.getLegalActions(iterCount%numAgent):
              sdot = s.generateSuccessor(iterCount%numAgent,a)
              result = min(result, _alphaBeta(sdot, iterCount+1, alpha, beta))
              beta = min(beta, result)
              if beta < alpha:
                break
            return result
          else: # Pacman Max
            result = -1e10
            for a in s.getLegalActions(iterCount%numAgent):
              sdot = s.generateSuccessor(iterCount%numAgent,a)
              result = max(result, _alphaBeta(sdot, iterCount+1, alpha, beta))
              alpha = max(alpha, result)
              if iterCount == 0:
                ActionScore.append(result)
              if beta < alpha:
                break
            return result

        result = _alphaBeta(gameState, 0, -1e20, 1e20)
        return gameState.getLegalActions(0)[ActionScore.index(max(ActionScore))]

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        numAgent = gameState.getNumAgents()
        ActionScore = []

        def _expectiMax(s, iterCount):
          if iterCount >= self.depth*numAgent or s.isWin() or s.isLose():
            return betterEvaluationFunction(s)
          if iterCount%numAgent != 0: #Ghost ave
            successorScore = []
            for a in s.getLegalActions(iterCount%numAgent):
              sdot = s.generateSuccessor(iterCount%numAgent,a)
              result = _expectiMax(sdot, iterCount+1)
              successorScore.append(result)
            averageScore = sum([ float(x)/len(successorScore) for x in successorScore])
            return averageScore
          else: # Pacman Max
            result = -1e10
            for a in s.getLegalActions(iterCount%numAgent):
              sdot = s.generateSuccessor(iterCount%numAgent,a)
              result = max(result, _expectiMax(sdot, iterCount+1))
              if iterCount == 0:
                ActionScore.append(result)
            return result
          
        result = _expectiMax(gameState, 0);
        return gameState.getLegalActions(0)[ActionScore.index(max(ActionScore))]

# def betterEvaluationFunction(currentGameState):
#     def _scoreFromGhost(gameState):
#       score = 0
#       for ghost in gameState.getGhostStates():
#         disGhost = manhattanDistance(gameState.getPacmanPosition(), ghost.getPosition())
#         if ghost.scaredTimer > 0:
#           score += pow(max(8 - disGhost, 0), 2)
#         else:
#           score -= pow(max(7 - disGhost, 0), 2)
#       return score

#     def _scoreFromFood(gameState):
#       disFood = []
#       for food in gameState.getFood().asList():
#         disFood.append(1.0/manhattanDistance(gameState.getPacmanPosition(), food))
#       if len(disFood)>0:
#         return max(disFood)
#       else:
#         return 0

#     def _scoreFromCapsules(gameState):
#       score = []
#       for Cap in gameState.getCapsules():
#         score.append(50.0/manhattanDistance(gameState.getPacmanPosition(), Cap))
#       if len(score) > 0:
#         return max(score)
#       else:
#         return 0

#     def _suicide(gameState):
#       score = 0
#       disGhost = 1e6
#       for ghost in gameState.getGhostStates():
#         disGhost = min(manhattanDistance(gameState.getPacmanPosition(), ghost.getPosition()), disGhost)
#       score -= pow(disGhost, 2)
#       if gameState.isLose():
#         score = 1e6
#       return score
#     score = currentGameState.getScore()
#     scoreGhosts = _scoreFromGhost(currentGameState)
#     scoreFood = _scoreFromFood(currentGameState)
#     scoreCapsules = _scoreFromCapsules(currentGameState)
#     return score + scoreGhosts + scoreFood + scoreCapsules

    #########################################
    #Function 2
# def betterEvaluationFunction(currentGameState):
#     newPos = currentGameState.getPacmanPosition()
#     newFood = currentGameState.getFood()
#     newGhostStates = currentGameState.getGhostStates()
#     newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

#     """Calculate distance to the nearest food"""
#     newFoodList = np.array(newFood.asList())
#     distanceToFood = [util.manhattanDistance(newPos, food) for food in newFoodList]
#     min_food_distance = 0
#     if len(newFoodList) > 0:
#         min_food_distance = distanceToFood[np.argmin(distanceToFood)]
#     """Calculate the distance to nearest ghost"""
#     ghostPositions = np.array(currentGameState.getGhostPositions())
#     distanceToGhost = [util.manhattanDistance(newPos, ghost) for ghost in ghostPositions]
#     min_ghost_distance = 0
#     nearestGhostScaredTime = 0
#     if len(ghostPositions) > 0:
#         min_ghost_distance = distanceToGhost[np.argmin(distanceToGhost)]
#         nearestGhostScaredTime = newScaredTimes[np.argmin(distanceToGhost)]
#         # avoid certain death
#         if min_ghost_distance <= 1 and nearestGhostScaredTime == 0:
#           return -999999
#         # eat a scared ghost
#         if min_ghost_distance <= 1 and nearestGhostScaredTime > 0:
#           return 999999

#     value = currentGameState.getScore() - min_food_distance
#     if nearestGhostScaredTime > 0:
#         # follow ghosts if scared
#         value -= min_ghost_distance
#     else:
#         value += min_ghost_distance
#     # print value
#     return value
  #################################################

    #Function 3
def betterEvaluationFunction(currentGameState):
    pos = currentGameState.getPacmanPosition() #position
    newScore = scoreEvaluationFunction(currentGameState)
    if currentGameState.isLose(): 
      return -float("inf")
    elif currentGameState.isWin():
      return float("inf")
    foodlist = currentGameState.getFood().asList() 
    manhattanDistanceToClosestFood = min(map(lambda x: util.manhattanDistance(pos, x), foodlist))
    distanceToClosestFood = manhattanDistanceToClosestFood
    numberOfCapsulesLeft = len(currentGameState.getCapsules()) # number of big dots
    numberOfFoodsLeft = len(foodlist)# number of foods left 
    scaredGhosts, activeGhosts = [], []
    for ghost in currentGameState.getGhostStates():
      if not ghost.scaredTimer:
        activeGhosts.append(ghost)
      else: 
        scaredGhosts.append(ghost)
    def getManhattanDistances(ghosts): 
      return map(lambda g: util.manhattanDistance(pos, g.getPosition()), ghosts)

    distanceToClosestActiveGhost = distanceToClosestScaredGhost = 0
    if activeGhosts:
      distanceToClosestActiveGhost = min(getManhattanDistances(activeGhosts))
    else: 
      distanceToClosestActiveGhost = float("inf")
    distanceToClosestActiveGhost = max(distanceToClosestActiveGhost, 5)
      
    if scaredGhosts:
      distanceToClosestScaredGhost = min(getManhattanDistances(scaredGhosts))
    else:
      distanceToClosestScaredGhost = 0

    score = 1   * newScore + \
            -1.5 * distanceToClosestFood + \
            -2    * (1./distanceToClosestActiveGhost) + \
            -2    * distanceToClosestScaredGhost + \
            -20 * numberOfCapsulesLeft + \
            -4    * numberOfFoodsLeft
    # print score
    return score
    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

