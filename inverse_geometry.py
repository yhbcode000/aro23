#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv, inv, norm, svd, eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits, jointlimitsviolated
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from setup_meshcat import updatevisuals

from tools import setcubeplacement


def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    oMcubeL = getcubeplacement(cube, LEFT_HOOK)  # placement of the left hand hook
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK)  # placement of the right hand hook

    left_frame_id = robot.model.getFrameId(LEFT_HAND)
    right_frame_id = robot.model.getFrameId(RIGHT_HAND)

    q = qcurrent
    success = False

    for _ in range(300):  # Number of iterations for convergence
        pin.framesForwardKinematics(robot.model, robot.data, q)
        pin.computeJointJacobians(robot.model, robot.data, q)
        oMframe_Left = robot.data.oMf[left_frame_id]
        oMframe_Right = robot.data.oMf[right_frame_id]

        # Calculate the error and Jacobian in position and orientation for both hands
        errL = pin.log(oMframe_Left.inverse() * oMcubeL).vector
        JL = pin.computeFrameJacobian(robot.model, robot.data, q, left_frame_id)  # [:3,:]
        errR = pin.log(oMframe_Right.inverse() * oMcubeR).vector
        JR = pin.computeFrameJacobian(robot.model, robot.data, q, right_frame_id)  # [:3,:]

        # Calculate the joint errors
        err = np.hstack([errL, errR])

        # Inverse each Jacobian and join them together

        JL_inv = pinv(JL)
        JR_inv = pinv(JR)

        jacobian = np.hstack([JL_inv, JR_inv])

        vq = jacobian @ err

        # Update the robot configuration
        q = pin.integrate(robot.model, q, vq)

        # Check if the current configuration is valid if valid stop the loop
        if norm(errL) < EPSILON and norm(errR) < EPSILON and not collision(robot, q) and not jointlimitsviolated(robot, q):
            success = True
            break

    if viz:
        updatevisuals(viz, robot, cube, q)

    return q, success


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals

    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()

    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz)

    updatevisuals(viz, robot, cube, q0)
