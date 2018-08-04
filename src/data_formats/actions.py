#!/usr/bin/env python
# -*- coding: utf-8 -*-

HUMAN_36M_ACTIONS = ["Directions",
           "Discussion",
           "Eating",
           "Greeting",
           "Phoning",
           "Photo",
           "Posing",
           "Purchases",
           "Sitting",
           "SittingDown",
           "Smoking",
           "Waiting",
           "WalkDog",
           "Walking",
           "WalkTogether"]

def define_actions(action):
    """
    :param action: specified action
    :return: a list of action(s)
    """
    if action == "All" or action == "all":
        return HUMAN_36M_ACTIONS

    if action not in HUMAN_36M_ACTIONS:
        raise (ValueError, "Unincluded action: {}".format(action))

    return [action]
