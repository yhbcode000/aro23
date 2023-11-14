#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv

from config import LEFT_HAND, RIGHT_HAND, LEFT_HOOK, RIGHT_HOOK, EPSILON
from pinocchio.utils import rotate
import time
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits, jointlimitsviolated
from setup_meshcat import updatevisuals


def random_cube():
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    z = np.random.uniform(-1, 1)
    # x = np.random.uniform(-1, 1)
    # y = np.random.uniform(-1, 1)
    # z = np.random.uniform(-1, 1)
    return pin.SE3(rotate('z', 0.), np.array([x, y, z]))


def is_valid_configuration(robot, q, hook_distance,cube):
    left_frame_id = robot.model.getFrameId(LEFT_HAND)
    right_frame_id = robot.model.getFrameId(RIGHT_HAND)
    oMframe_Left = robot.data.oMf[left_frame_id]
    oMframe_Right = robot.data.oMf[right_frame_id]
    hand_distance = np.linalg.norm(oMframe_Left.translation - oMframe_Right.translation)
    err = abs(hook_distance - hand_distance)
    if err < EPSILON:
        cube_position = (oMframe_Left.translation + oMframe_Right.translation) / 2
        cube_orientation = pin.SE3(oMframe_Left.rotation, cube_position).rotation
        oMcube = pin.SE3(cube_orientation, cube_position)
        setcubeplacement(robot,cube,oMcube)
    return not collision(robot, q) and not jointlimitsviolated(robot, q) and err< EPSILON and not collision_cube(cube)


def interpolate_configurations(q0, qe, discretionsteps=50):
    interpolated_path = []
    for step in range(1, discretionsteps + 1):
        alpha = step / discretionsteps
        interpolated_q = (1 - alpha) * q0 + alpha * qe
        interpolated_path.append(interpolated_q)

    return interpolated_path


def collision_cube(cube):
    pin.updateGeometryPlacements(cube.model, cube.data, cube.collision_model, cube.collision_data)
    return pin.computeCollisions(cube.collision_model, cube.collision_data, False)


# returns a collision free path from qinit to qgoal under grasping constraints
# the path is expressed as a list of configurations
def computepath(qinit, qgoal, cubeplacementq0, cubeplacementqgoal):
    tree = [(None, qinit)]
    path_found = False
    robot, cube, viz = setupwithmeshcat()

    while not path_found:
        # Sample a cube configuration
        cube_sample = random_cube()
        setcubeplacement(robot, cube, cube_sample)

        # Compute corresponding robot configuration
        q_sample, success = computeqgrasppose(robot, qinit, cube, cube_sample, viz)
        updatevisuals(viz, robot, cube, q_sample)
        if not success or collision(robot, q_sample) or jointlimitsviolated(robot, q_sample) or collision_cube(cube):
            continue  # Skip invalid configurations

        # Find nearest configuration in the tree
        q_nearest = min(tree, key=lambda node: np.linalg.norm(q_sample - node[1]))[1]

        # Interpolate between q_nearest and q_sample
        interpolated_path = interpolate_configurations(q_nearest, q_sample)

        oMcubeL = getcubeplacement(cube, LEFT_HOOK)  # placement of the left hand hook
        oMcubeR = getcubeplacement(cube, RIGHT_HOOK)
        hook_distance = np.linalg.norm(oMcubeL.translation - oMcubeR.translation)
        # Check each step in the interpolated path for validity
        for q in interpolated_path:
            if not is_valid_configuration(robot, q, hook_distance,cube):
                break  # Stop if any configuration in the path is invalid
        else:  # Python's for-else construct: else block runs if the loop wasn't 'broken'
            tree.append((q_nearest,q_sample))  # Add q_sample to the tree if the path is valid
        # if is_valid_configuration(robot, interpolated_path[0], cube):
        #     tree.append((q_nearest, interpolated_path[0]))
        #     for i in range(1, len(interpolated_path)):
        #         if is_valid_configuration(robot, interpolated_path[i], cube):
        #             tree.append((q_nearest, interpolated_path[i - 1]))
        #         else:
        #             break

        # Check if qgoal is reachable from any configuration in the tree
        interpolated_path_to_goal = interpolate_configurations(q_sample, qgoal)
        if any(is_valid_configuration(robot, q, hook_distance,cube) for q in
               interpolated_path_to_goal):
            # tree.append((q_nearest, interpolated_path_to_goal[0]))
            # for i in range(1, len(interpolated_path_to_goal)):
            #     tree.append((q_nearest, interpolated_path_to_goal[i - 1]))
            tree.append((q_sample, qgoal))
            path_found = True
        updatevisuals(viz, robot, cube, q_sample)

    # Backtrack to find the path from qinit to qgoal
    # ...
    current_config = qgoal
    path = []
    while current_config is not None:
        path.append(current_config)
        current_tuple = next((node for node in tree if np.array_equal(node[1], current_config)), None)
        if current_tuple is None:
            break
        current_config = current_tuple[0]
    path.append(qinit)
    print(tree)
    path.reverse()  # The path is constructed in reverse order, so reverse it
    print(path)
    return path


def displaypath(robot, path, dt, viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose

    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()
    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz)

    if not (successinit and successend):
        print("error: invalid initial or end configuration")

    path = computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    displaypath(robot, path, dt=5, viz=viz)  # you ll probably want to lower dt
