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
    x = np.random.uniform(-0.5, 0.5)
    y = np.random.uniform(-0.5, 0.5)
    z = np.random.uniform(0.9, 1.5)
    # x = np.random.uniform(-1, 1)
    # y = np.random.uniform(-1, 1)
    # z = np.random.uniform(-1, 1)
    return pin.SE3(rotate('z', 0.), np.array([x, y, z]))


# def is_valid_configuration(robot, q, hook_distance,cube):
#     left_frame_id = robot.model.getFrameId(LEFT_HAND)
#     right_frame_id = robot.model.getFrameId(RIGHT_HAND)
#     oMframe_Left = robot.data.oMf[left_frame_id]
#     oMframe_Right = robot.data.oMf[right_frame_id]
#     hand_distance = np.linalg.norm(oMframe_Left.translation - oMframe_Right.translation)
#     err = abs(hook_distance - hand_distance)
#     if err < EPSILON:
#         cube_position = (oMframe_Left.translation + oMframe_Right.translation) / 2
#         cube_orientation = pin.SE3(oMframe_Left.rotation, cube_position).rotation
#         oMcube = pin.SE3(cube_orientation, cube_position)
#         setcubeplacement(robot,cube,oMcube)
#     return not collision(robot, q) and not jointlimitsviolated(robot, q) and err< EPSILON and not collision_cube(cube)

def is_valid_configuration(robot,cube,cubeplacement,qinit):
    q, success = computeqgrasppose(robot, qinit, cube, cubeplacement)
    setcubeplacement(robot,cube,cubeplacement)
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)

    left_frame_id = robot.model.getFrameId(LEFT_HAND)
    right_frame_id = robot.model.getFrameId(RIGHT_HAND)
    oMframe_Left = robot.data.oMf[left_frame_id]
    oMframe_Right = robot.data.oMf[right_frame_id]
    oMcubeL = getcubeplacement(cube, LEFT_HOOK)  # placement of the left hand hook
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK)  # placement of the right hand hook
    errL = pin.log(oMframe_Left.inverse() * oMcubeL).vector
    errR = pin.log(oMframe_Right.inverse() * oMcubeR).vector
    # updatevisuals(viz, robot, cube, q)
    #time.sleep(0.01)
    if not success:
        return False,None
    return (not collision(robot, q) and not jointlimitsviolated(robot, q) and np.linalg.norm(errL) < EPSILON and not collision_cube(cube) and np.linalg.norm(errR) < EPSILON),q


def interpolate_configurations(q0, qe, discretionsteps=50):
    interpolated_path = []
    for step in range(1, discretionsteps + 1):
        alpha = step / discretionsteps
        interpolated_q = (1 - alpha) * q0 + alpha * qe
        interpolated_path.append(interpolated_q)

    return interpolated_path

def interpolate_cube_placement(start_placement, end_placement, steps=50):
    interpolated_placements = []

    for step in range(steps):
        alpha = step / float(steps - 1)

        # Interpolate translation
        interpolated_translation = (1 - alpha) * start_placement.translation + alpha * end_placement.translation

        # Interpolate rotation using slerp (spherical linear interpolation)
        #interpolated_rotation = start_placement.rotation.slerp(alpha, end_placement.rotation)

        # Construct the interpolated SE3 object
        interpolated_placement = pin.SE3(start_placement.rotation, interpolated_translation)
        interpolated_placements.append(interpolated_placement)

    return interpolated_placements

def collision_cube(cube):
    pin.updateGeometryPlacements(cube.model, cube.data, cube.collision_model, cube.collision_data)
    return pin.computeCollisions(cube.collision_model, cube.collision_data, False)

def check_validity_interpolated_path(path, robot, cube,qinit):
    successful_path = []
    success,current_config  = is_valid_configuration(robot,cube,path[0],qinit)
    if not success:
        return [],success
    for i in range(1,len(path)):
        success,q = is_valid_configuration(robot,cube,path[i],qinit)
        if success:
            successful_path.append([current_config,q,path[i]])
            current_config = q
        else:
            return successful_path,False
    return successful_path, True

# returns a collision free path from qinit to qgoal under grasping constraints
# the path is expressed as a list of configurations
def computepath(qinit, qgoal, cubeplacementq0, cubeplacementqgoal):
    tree = [[None, qinit,cubeplacementq0]]
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
        # updatevisuals(viz, robot, cube, q_sample)
        if not success or collision(robot, q_sample) or jointlimitsviolated(robot, q_sample) or collision_cube(cube):
            continue  # Skip invalid configurations

        # Find nearest configuration in the tree
        #q_nearest = min(tree, key=lambda node: np.linalg.norm(q_sample - node[1]))[1]
        #cube_nearest = min(tree, key=lambda node: np.linalg.norm(q_sample - node[1]))[2]
        _, nearest_index = kd_tree.query(q_sample)
        cube_nearest = tree[nearest_index][2]

        interpolated_cube_placement = interpolate_cube_placement(cube_nearest,cube_sample)

        interpolated_path, success = check_validity_interpolated_path(interpolated_cube_placement,robot,cube,qinit)

        if interpolated_path and success:
        #if interpolated_path:
            #tree.append([q_nearest,interpolated_path[0][0],interpolated_cube_placement[0]])
            for p in interpolated_path:
                tree.append(p)
                kd_tree = KDTree([node[1] for node in tree])

        interpolated_cube_placement_to_goal = interpolate_cube_placement(tree[-1][2],cubeplacementqgoal)
        interpolated_path_to_goal, reach_goal = check_validity_interpolated_path(interpolated_cube_placement_to_goal,robot,cube,qinit)

        # if interpolated_path_to_goal:
        #     #tree.append([tree[-1][1],interpolated_path_to_goal[0][0],interpolated_cube_placement_to_goal[0]])
        #     for p in interpolated_path_to_goal:
        #         tree.append(p)
        #     if reach_goal:
        #         path_found = True

        if interpolated_path_to_goal and reach_goal:
            for p in interpolated_path_to_goal:
                tree.append(p)
                kd_tree = KDTree([node[1] for node in tree])
            path_found = True
        # updatevisuals(viz, robot, cube, q_sample)

    # Backtrack to find the path from qinit to qgoal
    # ...
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
    # path.append(qinit) # fixed no need to add another qinit
    # print(tree)
    path.reverse()
    path.append(qgoal)
    # print(path)
    # for p in cube_positions:
    #     setcubeplacement(robot,cube,p)
    #     updatevisuals(viz,robot,cube,qinit)
    #     print(collision_cube(cube))
    #     time.sleep(0.01)
    #     if collision_cube(cube):
    #         break
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

    # collision_cube_placement = pin.SE3(rotate('z', 0.),np.array([0.33, -0.7, 0.93]))
    # setcubeplacement(robot,cube,collision_cube_placement)
    # updatevisuals(viz,robot,cube,q0)
    # print(collision_cube(cube))