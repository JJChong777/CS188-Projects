# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # init the dfs stack (the fringe) stack 是空的
    fringeStack = util.Stack()

    #


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # init the bfs queue (the fringe) queue 是空的
    queue = util.Queue()

    # make a visited dictionary for backtracking and avoid expanding already visited states 已访问的点
    visited = []

    # an empty list for the solution with state and direction
    pre_solution = []

    # make a empty list for the solution to be returned
    solution = []

    # backtrack dictionary
    backtrack = {}

    # insert the start state into the visited list as well
    visited.append(problem.getStartState())

    # pre_solution储存所有信息
    pre_solution.append(problem.getStartState())

    # for testing
    # counter = 0

    # get the relevant successor 下一步要访问的点
    # start_successors = problem.getSuccessors(problem.getStartState())
    for successor in problem.getSuccessors(problem.getStartState()):
        queue.push(successor)

    # check if the fringe is empty
    while not queue.isEmpty():
        # if counter == 20:
        #     break
        print(f"queue before pop: {queue.list}")
        node_to_explore = queue.pop()
        print(f"queue after pop: {queue.list}")

        while node_to_explore[0] in visited:
            # get the next node
            node_to_explore = queue.pop()
        if problem.isGoalState(node_to_explore[0]):
            print("\n\nSTOP!!!!!!!!!!!!!!!!!")
            # solution found
            for neighbor in problem.getSuccessors(node_to_explore[0]):
                print(f"Goal neighbor: {neighbor}")
                if neighbor[0] in visited:
                    backtrack[node_to_explore] = neighbor[0]
            # 开始回溯，构建solution：
            print(f"Goal: {node_to_explore[0]}")
            print(f"Final backtrack: {backtrack}")

            start_state = problem.getStartState()
            current_state = node_to_explore[0]
            while current_state != start_state:
                for (node, direction, _), prev_state in backtrack.items():
                    if node == current_state:
                        print("adding direction")
                        solution.append(direction)
                        current_state = prev_state
                        break
            solution.reverse()
            print(f"solution: {solution}")
            break
        # 不是目标，继续遍历
        # push the node to the visited list
        visited.append(node_to_explore[0])
        print(f"visited: {visited}")
        # for v_node in visited:
        for neighbor in problem.getSuccessors(node_to_explore[0]):
            print(f"neighbor: {neighbor}")
            if neighbor[0] in visited:
                backtrack[node_to_explore] = neighbor[0]
                print("add to back track")
        print(f"backtrack: {backtrack}")
        # find the successors of the node that we just explored

        next_successors = problem.getSuccessors(node_to_explore[0])
        print(f"next successors: {next_successors}\n\n")

        for next_successor in next_successors:
            if next_successor[0] not in visited:
                queue.push(next_successor)

        # counter += 1

    # testing
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    # util.raiseNotDefined()
    # 构建 solution 列表，忽略 pre_solution 中的第一个元素
    solution = [step[1] for step in pre_solution[1:]]

    # 输出 solution 列表
    print(f"at the end solution: {solution}")
    return solution
    # util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
