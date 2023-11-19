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
from setup_pinocchio import setuppinocchio
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits, jointlimitsviolated
from setup_meshcat import updatevisuals
from inverse_geometry import computeqgrasppose
import scipy


def random_cube():
    # Generate a random placement of the cube in an interval
    x = np.random.uniform(-0.5, 0.5)
    y = np.random.uniform(-0.5, 0.5)
    z = np.random.uniform(0.9, 1.5)
    return pin.SE3(rotate('z', 0.), np.array([x, y, z]))


def is_valid_configuration(robot, cube, cubeplacement, qinit):
    # check if the configuration is valid
    q, success = computeqgrasppose(robot, qinit, cube, cubeplacement)
    setcubeplacement(robot, cube, cubeplacement)
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)
    left_frame_id = robot.model.getFrameId(LEFT_HAND)
    right_frame_id = robot.model.getFrameId(RIGHT_HAND)
    oMframe_Left = robot.data.oMf[left_frame_id]
    oMframe_Right = robot.data.oMf[right_frame_id]
    oMcubeL = getcubeplacement(cube, LEFT_HOOK)
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK)
    errL = pin.log(oMframe_Left.inverse() * oMcubeL).vector
    errR = pin.log(oMframe_Right.inverse() * oMcubeR).vector
    if not success:
        return False, None
    return (not collision(robot, q) and not jointlimitsviolated(robot, q) and np.linalg.norm(
        errL) < EPSILON and not collision_cube(cube) and np.linalg.norm(errR) < EPSILON), q


def interpolate_cube_placement(start_placement, end_placement, steps=50):
    # Interpolate the placement of cube with discretion step of 50
    interpolated_placements = []

    for step in range(steps):
        alpha = step / float(steps - 1)

        # Interpolate translation
        interpolated_translation = (1 - alpha) * start_placement.translation + alpha * end_placement.translation

        # Construct the interpolated SE3 object
        interpolated_placement = pin.SE3(start_placement.rotation, interpolated_translation)
        interpolated_placements.append(interpolated_placement)

    return interpolated_placements


def collision_cube(cube):
    # Check if the cube is in collision
    pin.updateGeometryPlacements(cube.model, cube.data, cube.collision_model, cube.collision_data)
    return pin.computeCollisions(cube.collision_model, cube.collision_data, False)


def check_validity_interpolated_path(path, robot, cube, qinit):
    # Check the validity of a list of path and return the path if the path is valid
    successful_path = []
    success, current_config = is_valid_configuration(robot, cube, path[0], qinit)
    if not success:
        return [], success
    for i in range(1, len(path)):
        success, q = is_valid_configuration(robot, cube, path[i], qinit)
        if success:
            successful_path.append([current_config, q, path[i]])
            current_config = q
        else:
            return successful_path, False
    return successful_path, True


# returns a collision free path from qinit to qgoal under grasping constraints
# the path is expressed as a list of configurations
def computepath(qinit, qgoal, cubeplacementq0, cubeplacementqgoal):
    tree = [[None, qinit, cubeplacementq0]]
    path_found = False
    robot, table, obstacle, cube = setuppinocchio()
    KDTree = scipy.spatial.KDTree
    kd_tree = KDTree([qinit])
    while not path_found:
        # Sample a cube configuration
        cube_sample = random_cube()
        setcubeplacement(robot, cube, cube_sample)

        # Compute corresponding robot configuration
        q_sample, success = computeqgrasppose(robot, qinit, cube, cube_sample)
        if not success or collision(robot, q_sample) or jointlimitsviolated(robot, q_sample) or collision_cube(cube):
            continue  # Skip invalid configurations

        # Find nearest configuration in the tree
        _, nearest_index = kd_tree.query(q_sample)
        cube_nearest = tree[nearest_index][2]

        # Interpolate the cube palcement to get a list of cube placements
        interpolated_cube_placement = interpolate_cube_placement(cube_nearest, cube_sample)

        # Check the validity of the interpolated path and add the path of configurations to our tree if the path is
        # correct
        interpolated_path, success = check_validity_interpolated_path(interpolated_cube_placement, robot, cube, qinit)

        if interpolated_path and success:
            print("One Successful Interpolated Path Found")
            for p in interpolated_path:
                tree.append(p)
                kd_tree = KDTree([node[1] for node in tree])

        # Interpolate the placement between the latest cube placement and the goal cube placement.
        # If this path is valid, add the path to the tree and  break the while loop
        interpolated_cube_placement_to_goal = interpolate_cube_placement(tree[-1][2], cubeplacementqgoal)
        interpolated_path_to_goal, reach_goal = check_validity_interpolated_path(interpolated_cube_placement_to_goal,
                                                                                 robot, cube, qinit)

        if interpolated_path_to_goal and reach_goal:
            for p in interpolated_path_to_goal:
                tree.append(p)
                kd_tree = KDTree([node[1] for node in tree])
            path_found = True


    # Backtrack to find the path from qinit to qgoal
    current_config = tree[-1][1]
    path = []
    cube_positions = []
    while current_config is not None:
        path.append(current_config)
        current_tuple = next((node for node in tree if np.array_equal(node[1], current_config)), None)
        if current_tuple is None:
            break
        current_config = current_tuple[0]
        cube_positions.append(current_tuple[2])
    path.reverse()
    path.append(qgoal)
    return path


def displaypath(robot, path, dt, viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()
    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz)

    if not (successinit and successend):
        print("error: invalid initial or end configuration")

    path = computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    displaypath(robot, path, dt=0.01, viz=viz)  # you ll probably want to lower dt

