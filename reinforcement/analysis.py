# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.


# key concepts:

# answerDiscount causes the agent to prefer different kind of rewards
# low discount means it prefers shorter term rewards, higher discount means it prefers longer term rewards
# more steps agent takes -> more heavily discounted rewards -> agent is pushed to take shorter term rewards

# answerNoise causes the agent to prefer different risk levels of paths
# low noise means it can trust its moves, so it will take the riskier path
# higher noise means that the agent cannot trust its moves, making it take the less risky path


# answerLivingReward (reward per step) causes the agent to favor exploring vs favoring the shorter path
# positive living reward -> rewarded for exploring (transitioning from one state to another) -> explore more
# negative living reward -> penalized for exploring -> agent motivated to reach terminal state faster


def question2a():
    """
    Prefer the close exit (+1), risking the cliff (-10).
    """
    answerDiscount = 0.2
    answerNoise = 0
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question2b():
    """
    Prefer the close exit (+1), but avoiding the cliff (-10).
    """
    answerDiscount = 0.4
    answerNoise = 0.3
    answerLivingReward = -0.1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question2c():
    """
    Prefer the distant exit (+10), risking the cliff (-10).
    """
    answerDiscount = 0.8
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question2d():
    """
    Prefer the distant exit (+10), avoiding the cliff (-10).
    """
    answerDiscount = 0.8
    answerNoise = 0.5
    answerLivingReward = 0.1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question2e():
    """
    Avoid both exits and the cliff (so an episode should never terminate).
    """
    answerDiscount = 0
    answerNoise = 0
    answerLivingReward = 100
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


if __name__ == "__main__":
    print("Answers to analysis questions:")
    import analysis

    for q in [q for q in dir(analysis) if q.startswith("question")]:
        response = getattr(analysis, q)()
        print("  Question %s:\t%s" % (q, str(response)))
