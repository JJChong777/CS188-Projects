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
    # init the closed set to check if node visited
    closed = set()

    # init the dfs stack (the fringe) 是空的
    fringeStack = util.Stack()

    # just putting the start state here for convenience
    start_state = problem.getStartState()

    # also insert the start state into the fringe
    closed.add(start_state)

    # insert the neighbours of the initial state into the fringe
    # also insert a list (the path required to get to the node)
    for child_node in problem.getSuccessors(start_state):
        fringeStack.push([child_node, [child_node[1]]])

    while not fringeStack.isEmpty():
        node = fringeStack.pop()

        # return the solution
        if problem.isGoalState(node[0][0]):
            return node[1]

        if node[0][0] not in closed:
            closed.add(node[0][0])
            for child_node in problem.getSuccessors(node[0][0]):
                fringeStack.push([child_node, node[1] + [child_node[1]]])
        # print(f"closed: {closed}")
        # print(f"fringe stack: {fringeStack.list}\n")


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # init the closed set to check if node visited
    closed = set()

    # init the bfs queue (the fringe) 是空的
    fringeQueue = util.Queue()

    # just putting the start state here for convenience
    start_state = problem.getStartState()

    # insert the start state into the fringe
    closed.add(start_state)

    # insert the neighbours of the initial state into the fringe
    # also insert a list (the path required to get to the node)
    for child_node in problem.getSuccessors(start_state):
        fringeQueue.push([child_node, [child_node[1]]])

    while not fringeQueue.isEmpty():
        node = fringeQueue.pop()

        # return the solution
        if problem.isGoalState(node[0][0]):
            return node[1]

        if node[0][0] not in closed:
            closed.add(node[0][0])
            for child_node in problem.getSuccessors(node[0][0]):
                fringeQueue.push([child_node, node[1] + [child_node[1]]])
        # print(f"closed: {closed}")
        # print(f"fringe queue: {fringeQueue.list}\n")


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # init the closed set to check if node visited
    closed = set()

    # init the priority queue (the fringe) 是空的
    fringePrioQueue = util.PriorityQueue()

    # just putting the start state here for convenience
    start_state = problem.getStartState()

    # insert the start state into the fringe
    closed.add(start_state)

    # insert the neighbours of the initial state into the fringe
    # also insert a list (the path required to get to the node)
    for child_node in problem.getSuccessors(start_state):
        fringePrioQueue.update(
            [child_node, [child_node[1]]], problem.getCostOfActions([child_node[1]])
        )

    # when there are neighbors
    while not fringePrioQueue.isEmpty():
        node = fringePrioQueue.pop()

        # return the solution
        if problem.isGoalState(node[0][0]):
            # return the cost
            return node[1]

        # if the node has never been visited, add it to visited list (closed)
        # and update its neighbors' cost
        if node[0][0] not in closed:
            closed.add(node[0][0])

            for child_node in problem.getSuccessors(node[0][0]):
                fringePrioQueue.update(
                    [child_node, node[1] + [child_node[1]]],
                    problem.getCostOfActions(node[1] + [child_node[1]]),
                )
        # print(f"closed: {closed}")
        # print(f"fringe prio queue: {fringePrioQueueWithFunction.list}\n")


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # init the closed set to check if node visited
    closed = set()

    # init the priority queue (the fringe) 是空的
    fringePrioQueue = util.PriorityQueue()

    # just putting the start state here for convenience
    start_state = problem.getStartState()

    # insert the start state into the fringe
    closed.add(start_state)

    # insert the neighbours of the initial state into the fringe
    # also insert a list (the path required to get to the node)
    for child_node in problem.getSuccessors(start_state):
        fringePrioQueue.update(
            [child_node, [child_node[1]]],
            problem.getCostOfActions([child_node[1]])
            + heuristic(child_node[0], problem),
        )

    while not fringePrioQueue.isEmpty():
        node = fringePrioQueue.pop()

        # return the solution
        if problem.isGoalState(node[0][0]):
            return node[1]

        if node[0][0] not in closed:
            closed.add(node[0][0])
            for child_node in problem.getSuccessors(node[0][0]):
                fringePrioQueue.update(
                    [child_node, node[1] + [child_node[1]]],
                    problem.getCostOfActions(node[1] + [child_node[1]])
                    + heuristic(child_node[0], problem),
                )
        # print(f"closed: {closed}")
        # print(f"fringe prio queue: {fringePrioQueueWithFunction.list}\n")


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
