# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state, action, nextState)
            mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
        Run the value iteration algorithm. Note that in standard
        value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            # get the copy of values to run value iteration on so that it can be calculated based on the previous state
            newValues = self.values.copy()
            # iterate through all the states
            for curState in self.mdp.getStates():
                # if terminal, qValue of state is 0
                if self.mdp.isTerminal(curState):
                    newValues[curState] = 0
                else:
                    # init a list for the action values for the curState
                    actionValues = []
                    # iterate through all actions in current state
                    for posAction in self.mdp.getPossibleActions(curState):
                        # find all of the possible qValues for that state
                        actionValue = self.computeQValueFromValues(curState, posAction)
                        # and append it to the list
                        actionValues.append(actionValue)
                    # if the list of actionValues are not empty
                    if actionValues:
                        # assign the max of the actionValues to that state
                        newValues[curState] = max(actionValues)
            # replace the values counter with the new one
            self.values = newValues

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # init q value to 0
        qValue = 0
        # for all the nextStates and the prob for that state given that action
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            # get the reward of the state, action, nextState R(s,a,s′)
            reward = self.mdp.getReward(state, action, nextState)
            # get the qValue using the formula (∑s' T(s,a,s')[R(s,a,s')+γVk(s')])
            # γ = self.discount, s' = nextState, s = curState, a = action, R = reward, Vk(s') = previous value of the next state, T(s,a,s') = prob
            qValue += prob * (reward + (self.discount * self.getValue(nextState)))
        return qValue

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # no legal actions, return None
        if self.mdp.isTerminal(state):
            return None
        # init bestAction and bestQValue
        bestAction = None
        bestQValue = float("-inf")
        # loop through all the actions
        for action in self.mdp.getPossibleActions(state):
            # get the curQValue for that action
            curQValue = self.getQValue(state, action)
            # if higher QValue for action found, assign the bestQValue and bestAction to it
            if curQValue > bestQValue:
                bestQValue = curQValue
                bestAction = action
        # return the best action among all of the actions for that state
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
