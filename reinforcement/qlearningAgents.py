# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import nn
import backend
import gridworld


import random, util, math
import numpy as np
import copy


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent
    Functions you should fill in:
      - computeValueFromQValues
      - computeActionFromQValues
      - getQValue
      - getAction
      - update
    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)
    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # keep track of all the QValues for every (state, action)
        self.values = util.Counter()

    def getQValue(self, state, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state
        or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # get back the QValue from self.values
        # since Counter initializes all values to 0.0, it returns 0.0 if we have never seen the (state, action)
        return self.values[(state, action)]

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        actionValues = []
        for legalAction in legalActions:
            actionValues.append(self.getQValue(state, legalAction))
        return max(actionValues)

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        # if no legalActions return None
        if not legalActions:
            return None
        # keep a list of actions with the bestQValue to break ties randomly
        bestActions = []
        bestQValue = float("-inf")
        for legalAction in legalActions:
            curQValue = self.getQValue(state, legalAction)
            if curQValue > bestQValue:
                bestQValue = curQValue
                bestActions = [legalAction]
            # step to ensure that all actions with the bestQValue are kept in the list
            if curQValue == bestQValue:
                bestActions.append(legalAction)
        # break the tie randomly
        return random.choice(bestActions)

    def getAction(self, state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.
        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # if we get epsilon probability, return a random action
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        # else, return the best action according to the Q Values
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward: float):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here
        NOTE: You should never call this function,
        it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # sample = R(s,a,s') + γ max_a' Q(s',a')
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        # Q(s,a) ← (1−α)Q(s,a) + α * sample
        self.values[(state, action)] = (1 - self.alpha) * self.getQValue(
            state, action
        ) + self.alpha * sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args["epsilon"] = epsilon
        args["gamma"] = gamma
        args["alpha"] = alpha
        args["numTraining"] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
    ApproximateQLearningAgent
    You should only have to overwrite getQValue
    and update.  All other QLearningAgent functions
    should work as is.
    """

    def __init__(self, extractor="IdentityExtractor", **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # take a look at featureExtractors.py
        features = self.featExtractor.getFeatures(state, action)
        # features is now a dictionary with the key value pair of the feature and feature value respectively
        # Q(s,a)= ∑(i=1 to n) f_i(s,a) w_i
        # there are 1 to n features, f_i(s,a) is the feature value for the ith feature, w_i is the weight assigned to that ith feature

        qValue = sum(
            featureValue * self.weights[feature]
            for feature, featureValue in features.items()
        )

        return qValue

    def update(self, state, action, nextState, reward: float):
        """
        Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # difference = (r + γ max_a'Q(s', a')) − Q(s,a)
        # r = reward, γ = discount factor, max_a'Q(s', a') = self.computeValueFromQValues(nextState), Q(s,a) = self.getQValue(state, action)
        difference = (
            reward + self.discount * self.computeValueFromQValues(nextState)
        ) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for feature, featureValue in features.items():
            # w_i ← w_i + α * difference * f_i(s,a)
            self.weights[feature] = self.weights[feature] + (
                self.alpha * difference * featureValue
            )

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
