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
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states 获取当前状态下的所有合法动作
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        # 为每个合法动作计算评估分数。
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        # 找到最高的评估分数。
        bestScore = max(scores)
        # 找到所有具有最高分数的动作索引
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        # 随机选择一个具有最高分数的动作。
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        # 返回选择的动作
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # 生成执行给定动作后的下一个状态
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # 获取 Pacman 在邻居中的新位置
        newPos = successorGameState.getPacmanPosition()
        # 获取继承状态中的食物布置
        newFood = successorGameState.getFood()
        # 获取下一个状态中的幽灵状态
        newGhostStates = successorGameState.getGhostStates()
        # 获取每个幽灵的受惊吓时间
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # 列表形式的食物
        newFoodList = newFood.asList()
        # 获取邻居状态的得分
        score = successorGameState.getScore()
        ghostPositions = successorGameState.getGhostPositions()
        foodDistList = []
        for foodPos in newFoodList:
            if newFood[foodPos[0]][foodPos[1]] == True:
                foodDist = manhattanDistance(foodPos, newPos)
                if foodDist == 0:
                    foodDistList.append(1)
                else:
                    foodDistList.append(1 / manhattanDistance(foodPos, newPos))
        sumFoodDist = sum(foodDistList)

        # find the manhattan distance from pacman's position to the nearest ghost
        ghostDistList = []
        for ghostPos in ghostPositions:
            ghostDist = manhattanDistance(ghostPos, newPos)
            if ghostDist == 0:
                ghostDistList.append(0)
            else:
                ghostDistList.append(1 / manhattanDistance(ghostPos, newPos))
        sumGhostDist = sum(ghostDistList)
        return 0.7 * score + (sumFoodDist) - 2 * (sumGhostDist)


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # print("get to getAction fgunction!")

        def value(state: GameState, depth, agentIndex):
            # print(f"isWin: {state.isWin()}")
            # print(f"isLose: {state.isLose()}")
            # print(f"depth: {depth}")
            # Terminal state or depth limit reached
            if state.isWin() or state.isLose() or depth == self.depth:
                # 若满足终止条件，即胜利、失败或达到最大深度，返回评估值
                # If the termination condition is met, return evaluation value
                return self.evaluationFunction(state)

            # return the max value for pacman
            if agentIndex == 0:
                return max_value(state, depth)

            # return the min value for the ghosts
            if agentIndex > 0:
                return min_value(state, depth, agentIndex)

        # pacman's function
        def max_value(state: GameState, depth):
            v = -float("inf")
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                v = max(v, value(successor, depth, 1))
            # print(f"max v: {v}")
            return v

        # minimizing function for ghosts
        def min_value(state: GameState, depth, agentIndex):
            v = float("inf")
            actions = state.getLegalActions(agentIndex)
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                # 检查当前代理是否是最后一个代理（最后一个鬼）
                # if this is the last ghost
                # increment depth by 1
                if agentIndex == state.getNumAgents() - 1:
                    v = min(v, value(successor, depth + 1, 0))
                # find the minimum v that can be obtained by this ghost
                else:
                    v = min(v, value(successor, depth, agentIndex + 1))
            # print(f"min v: {v}")
            return v

        # Get the best action for Pacman
        legalMoves = gameState.getLegalActions(0)
        # print(f"legal moves: {legalMoves}")
        bestScore = float("-inf")
        bestAction = None

        # for every action
        for action in legalMoves:
            # get neighbors state of the action
            successor = gameState.generateSuccessor(0, action)
            # expected score
            score = value(successor, 0, 1)
            # if the expected score is bigger than the current best score,
            # update best score and action
            # 如果该动作的预期价值比当前最佳值大，更新最佳值和最佳动作
            # print(f"current score: {score}")
            if score > bestScore:
                bestScore = score
                bestAction = action
                # print(f"new best score: {score}")
        # print(f"best score: {bestScore}")
        # print(f"best action: {bestAction}")
        return bestAction
        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # pacman's function, calculate the max value of ghosts layer
        def max_value(state: GameState, depth, alpha, beta):
            # if we win / lose / meet the deepest layer, return directly
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            # initialize
            v = -float("inf")

            # for all valid actions
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)

                # calculate min value, update v
                v = max(v, min_value(successor, depth, 1, alpha, beta))

                # if true, cut it
                if v > beta:
                    return v 
                
                # else: update alpha
                alpha = max(alpha, v)
            return v

        # minimizing function for ghosts
        def min_value(state: GameState, depth, agentIndex, alpha, beta):

            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            # initialize
            v = float("inf")

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)

                # when it is the last ghost, go back to calculate pacman's value (max_value)
                if agentIndex == state.getNumAgents() - 1:
                    v = min(v, max_value(successor, depth + 1, alpha, beta))
                else:
                    # else, keep going to find min value
                    v = min(v, min_value(successor, depth, agentIndex + 1, alpha, beta))

                # when it is true, cut it
                if v < alpha:
                    return v  

                # update beta
                beta = min(beta, v)
            return v

        # initialize
        bestScore = float("-inf")
        bestAction = None
        alpha = -float("inf")
        beta = float("inf")

        # for all valid actions
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = min_value(successor, 0, 1, alpha, beta)

            # if current score is better than bestScore, update
            if score > bestScore:
                bestScore = score
                bestAction = action
            
            # update alpha
            alpha = max(alpha, bestScore)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # Goal: choose an action that can maximize expected return
        # 选择一个能最大化其期望收益的动作
        def value(state, depth, agentIndex):
            # use depth to limit the max searching depth
            # use agentIndex to identify who is moving
            # 计算每个动作的预期价值

            # Terminal state or depth limit reached
            if state.isWin() or state.isLose() or depth == self.depth:
                # 若满足终止条件，即胜利、失败或达到最大深度，返回评估值
                # If the termination condition is met, return evaluation value
                return self.evaluationFunction(state)

            # For Pacman: max_value
            if agentIndex == 0:
                # 如果到达了最大深度，则返回此函数的值
                return max_value(state, depth)

            # For Ghosts: expected_value
            else:
                # 如果到达了终端状态（赢或输）则返回此函数的值
                return exp_value(state, depth, agentIndex)

        # Pacman:
        def max_value(state, depth):
            # 计算当前状态下的最大值

            # initialize maximal(alpha) to negative infinity 初始化最大值为负无穷
            v = float("-inf")
            for action in state.getLegalActions(0):
                # 遍历所有合法动作，生成继承状态
                # for every action, get neighbors (successors)
                successor = state.generateSuccessor(0, action)
                # 递归调用 value 函数来获取继承状态的值。
                # update the maximal (alpha)
                v = max(v, value(successor, depth, 1))
            return v

        # Goast:
        def exp_value(state, depth, agentIndex):
            # initialize expected value v
            v = 0
            # get all actions
            actions = state.getLegalActions(agentIndex)
            # 计算每个动作的概率，这里每个动作的概率相等?
            # calculate probability of actions
            p = 1.0 / len(actions)

            for action in actions:
                # 获取邻居信息
                # get neighbors
                successor = state.generateSuccessor(agentIndex, action)
                # 检查当前代理是否是最后一个代理（最后一个鬼）
                # if this is the last ghost
                if agentIndex == state.getNumAgents() - 1:  # Last ghost
                    # 递归调用 value 函数获取继承状态的值，并乘以动作的概率 p；
                    # 向下一层搜索
                    # update v
                    v += p * value(successor, depth + 1, 0)
                else:
                    # 递归调用 value 函数来获取继承状态的值
                    # update v
                    v += p * value(successor, depth, agentIndex + 1)
            return v

        # Get the best action for Pacman
        legalMoves = gameState.getLegalActions(0)
        bestScore = float("-inf")
        bestAction = None

        # for every action
        for action in legalMoves:
            # get neighbors state of the action
            successor = gameState.generateSuccessor(0, action)
            # expected score
            score = value(successor, 0, 1)
            # if the expected score is bigger than the current best score,
            # update best score and action
            # 如果该动作的预期价值比当前最佳值大，更新最佳值和最佳动作
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Instead of using manhattan distance, use the distance calculated using BFS that also considers the walls
    """
    "*** YOUR CODE HERE ***"
    # print("get to betterEvaluationFunction!")
    newFood = currentGameState.getFood()
    newFoodList = newFood.asList()
    score = currentGameState.getScore()
    ghostPositions = currentGameState.getGhostPositions()
    newPos = currentGameState.getPacmanPosition()
    walls = currentGameState.getWalls()

    # print(f"ghost positions: {ghostPositions}")
    # print(f"newPos: {newPos}")
    # print(f"newFoodList: {newFoodList}")

    def path_length(start, end, walls):

        rows = walls.width
        cols = walls.height

        visited = set([start])

        movements = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        queue = util.Queue()

        queue.push((start, 0))

        while not queue.isEmpty():
            point, distance = queue.pop()
            if point == end:
                return distance
            for move in movements:
                neighbor = (point[0] + move[0], point[1] + move[1])
                # print(f"neighbor: {neighbor}")
                # breakpoint()
                if (
                    neighbor not in visited
                    and not walls[neighbor[0]][neighbor[1]]
                    and 0 <= neighbor[0] < rows
                    and 0 <= neighbor[1] < cols
                ):
                    queue.push((neighbor, distance + 1))
                    visited.add(neighbor)

        return 9999999

    foodDistList = []
    for foodPos in newFoodList:
        if newFood[foodPos[0]][foodPos[1]] == True:
            foodDist = path_length(foodPos, newPos, walls)
            # print(f"food dist: {foodDist}")
            if foodDist == 0:
                foodDistList.append(1)
            else:
                foodDistList.append(1 / path_length(foodPos, newPos, walls))
    sumFoodDist = sum(foodDistList)

    # find the manhattan distance from pacman's position to the nearest ghost
    ghostDistList = []
    for ghostPos in ghostPositions:
        ghostPos = (int(ghostPos[0]), int(ghostPos[1]))
        ghostDist = path_length(ghostPos, newPos, walls)
        # print(f"ghost dist: {ghostDist}")
        if ghostDist == 0:
            ghostDistList.append(0)
        else:
            ghostDistList.append(1 / path_length(ghostPos, newPos, walls))
    sumGhostDist = sum(ghostDistList)
    return 2 * score + (sumFoodDist) - (sumGhostDist)

    # util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
