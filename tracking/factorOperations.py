# factorOperations.py
# -------------------
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

from typing import List
from bayesNet import Factor
import functools
from util import raiseNotDefined

# extra import for question 2
import itertools


def joinFactorsByVariableWithCallTracking(callTrackingList=None):

    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that
        contain that variable.

        Returns a tuple of
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(("join", joinVariable))

        currentFactorsToJoin = [
            factor for factor in factors if joinVariable in factor.variablesSet()
        ]
        currentFactorsNotToJoin = [
            factor for factor in factors if joinVariable not in factor.variablesSet()
        ]

        # typecheck portion
        numVariableOnLeft = len(
            [
                factor
                for factor in currentFactorsToJoin
                if joinVariable in factor.unconditionedVariables()
            ]
        )
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factor)
            raise ValueError(
                "The joinBy variable can only appear in one factor as an \nunconditioned variable. \n"
                + "joinVariable: "
                + str(joinVariable)
                + "\n"
                + ", ".join(
                    map(
                        str,
                        [
                            factor.unconditionedVariables()
                            for factor in currentFactorsToJoin
                        ],
                    )
                )
            )

        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable


joinFactorsByVariable = joinFactorsByVariableWithCallTracking()

########### ########### ###########
########### QUESTION 2  ###########
########### ########### ###########


def joinFactors(factors: List[Factor]):
    """
    Input factors is a list of factors.

    You should calculate the set of unconditioned variables and conditioned
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input
    (such as getProbability and setProbability) can handle
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError(
                "unconditionedVariables can only appear in one factor. \n"
                + "unconditionedVariables: "
                + str(intersect)
                + "\nappear in more than one input factor.\n"
                + "Input factors: \n"
                + "\n".join(map(str, factors))
            )

    "*** YOUR CODE HERE ***"
    # joinedFactor = Factor()
    joinUncondVar = set()

    # test print to see what factors look like
    for index, factor in enumerate(factors):
        # print(f"Factor {index+1}")
        # print(
        #     f"getAllPossibleAssignmentDicts: {factor.getAllPossibleAssignmentDicts()}"
        # )
        # print(f"getProbability: {factor.getProbability({'W': 'sun', 'D': 'wet'})}")
        # print(f"setProbability: {factor.setProbability()}")

        # this is the conditioned variable e.g. P(X), X is the unconditioned variable
        # I think this is also for P(X|Y), X is the unconditioned variable
        # print(f"unconditionedVariables: {factor.unconditionedVariables()}")

        # this is the conditioned variable e.g. P(X|Y), Y is the conditioned variable
        # print(f"conditionedVariables: {factor.conditionedVariables()}")

        # print(f"variableDomainsDict: {factor.variableDomainsDict()}\n")

        # put all the unconditioned variables in a joinUncondVar set
        for uncondtionedVar in factor.unconditionedVariables():
            # print(f"index {index} uncondtionedVar: {uncondtionedVar}")
            joinUncondVar.add(uncondtionedVar)
        # print(f"joinUncondVar: {joinUncondVar}")

        # check for all the remaining variables and add them to the condtioned variable dict at the last factor
        if index == len(factors) - 1:
            for uncondtionedVar in factor.unconditionedVariables():
                # print(f"index {index} uncondtionedVar: {uncondtionedVar}")
                joinUncondVar.add(uncondtionedVar)
            allVar = list(factor.variableDomainsDict().keys())
            joinCondVar = set(set(allVar) - joinUncondVar)
            # print(f"joinCondVar: {joinCondVar}")
            allDomains = [factor.variableDomainsDict()[var] for var in allVar]
            allCombinations = list(itertools.product(*allDomains))
            # Convert the combinations to a list of dictionaries
            allPossibleAssignments = []
            for combination in allCombinations:
                assignment = {allVar[i]: combination[i] for i in range(len(allVar))}
                allPossibleAssignments.append(assignment)
            # print(f"allPossibleAssignments: {allPossibleAssignments}")
            # create the new factor to start assigning probabilities to it
            joinFactor = Factor(
                joinUncondVar, joinCondVar, factor.variableDomainsDict()
            )
            for assignment in allPossibleAssignments:
                joinProbability = 1
                for factor in factors:
                    joinProbability *= factor.getProbability(assignment)
                # print(f"probability for {assignment} is {joinProbability}")
                joinFactor.setProbability(assignment, joinProbability)

            return joinFactor

    # the actual code
    "*** END YOUR CODE HERE ***"


########### ########### ###########
########### QUESTION 3  ###########
########### ########### ###########


def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor: Factor, eliminationVariable: str):
        """
        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.

        You should calculate the set of unconditioned variables and conditioned
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(("eliminate", eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError(
                "Elimination variable is not an unconditioned variable "
                + "in this factor\n"
                + "eliminationVariable: "
                + str(eliminationVariable)
                + "\nunconditionedVariables:"
                + str(factor.unconditionedVariables())
            )

        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError(
                "Factor has only one unconditioned variable, so you "
                + "can't eliminate \nthat variable.\n"
                + "eliminationVariable:"
                + str(eliminationVariable)
                + "\n"
                + "unconditionedVariables: "
                + str(factor.unconditionedVariables())
            )

        "*** YOUR CODE HERE ***"
        # make a copy of the unconditioned vars
        remainUncondVars = set(factor.unconditionedVariables())

        # elimination variable must be a unconditioned variable
        remainUncondVars.remove(eliminationVariable)

        # remaining variables of the factor
        remainVars = list(remainUncondVars.union(factor.conditionedVariables()))

        remainVarDomainDict = result_dict = {
            var: domain
            for var, domain in factor.variableDomainsDict().items()
            if var in remainVars
        }

        # print(f"remainUncondVars: {remainUncondVars}")
        # print(f"remainCondVars: {factor.conditionedVariables()}")
        # print(f"remainVarDomainDict: {remainVarDomainDict}")

        resultFactor = Factor(
            remainUncondVars, factor.conditionedVariables(), remainVarDomainDict
        )

        resultAssignmentDicts = resultFactor.getAllPossibleAssignmentDicts()
        # print(f"resultAssignmentDict: {resultAssignmentDicts}")

        origAssignmentDicts = factor.getAllPossibleAssignmentDicts()

        for resultAssign in resultAssignmentDicts:
            # init the resulting probability sum
            resultProbability = 0
            for origAssign in origAssignmentDicts:
                if all(origAssign[var] == resultAssign[var] for var in resultAssign):
                    prob = factor.getProbability(origAssign)
                    resultProbability += prob
            # Output the resulting probability
            # print(f"Resulting probability for {resultAssign}: {resultProbability}")
            resultFactor.setProbability(resultAssign, resultProbability)

        return resultFactor
        # for assignment in resultAssignmentDicts:
        #     key, value = next(iter())
        # find all the combinations
        # allCombinations = list(
        #     itertools.product(
        #         *[factor.variableDomainsDict()[var] for var in remainVars]
        #     )
        # )

        # print(allCombinations)

        # print(factor.variableDomainsDict())
        # elimFactor = [
        #     {remainVars[i]: combination[i] for i in range(len(remainVars))}
        #     for combination in allCombinations
        # ]
        # print(result)

        # resultFactor = Factor(
        #     remainUncondVars,
        #     factor.conditionedVariables(),
        # )

        # return
        # Use a set to collect unique combinations of W and D

        # for assignment in factor.getAllPossibleAssignmentDicts():

        # raiseNotDefined()
        "*** END YOUR CODE HERE ***"

    return eliminate


eliminate = eliminateWithCallTracking()
