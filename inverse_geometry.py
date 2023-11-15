#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement

def computeqgrasppose(robot, qcurrent, cube, cubetarget):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    oMcubeL = getcubeplacement(cube, LEFT_HOOK)  # placement of the left hand hook
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK)  # placement of the right hand hook

    left_frame_id = robot.model.getFrameId(LEFT_HAND)
    right_frame_id = robot.model.getFrameId(RIGHT_HAND)

    q = qcurrent
    success = False
    q_bias = robot.q0
    alpha = 0.01
    vq_bias = alpha * (q_bias - q)

    for _ in range(300):  # Number of iterations for convergence
        pin.framesForwardKinematics(robot.model, robot.data, q)
        pin.computeJointJacobians(robot.model, robot.data, q)
        oMframe_Left = robot.data.oMf[left_frame_id]
        oMframe_Right = robot.data.oMf[right_frame_id]

        # Calculate the error in position and orientation for both hands
        errL = pin.log(oMframe_Left.inverse() * oMcubeL).vector
        JL = pin.computeFrameJacobian(robot.model, robot.data, q, left_frame_id)  # [:3,:]
        errR = pin.log(oMframe_Right.inverse() * oMcubeR).vector
        JR = pin.computeFrameJacobian(robot.model, robot.data, q, right_frame_id)  # [:3,:]

        err = np.hstack([errL,errR])
        #print(err.shape)

        # Compute the Jacobians for both hands

        JL_inv = pinv(JL)
        JR_inv = pinv(JR)



        jacobian = np.hstack([JL_inv,JR_inv])

        # Compute the velocity vectors (dq) for both hands
        # dqL = -pinv(JL) @ errL
        # dqR = -pinv(JR) @ errR

        #print(jacobian.shape)
        vq = jacobian @ err
        #vq += vq_bias
        #vq = np.hstack((vqL,vqR))
        # Update the robot configuration
        q = pin.integrate(robot.model,q, vq)

        # Project the configuration to joint limits if necessary
        #q = projecttojointlimits(robot,q)

        # Check for collisions, if any, break and return failure
        if norm(errL) < EPSILON and norm(errR) < EPSILON and not collision(robot, q):
            success = True
            break
        
    # # By default the success is false.
    # # If error is below a threshold, stop the iteration
    # if not (norm(errL) < EPSILON and norm(errR) < EPSILON and not collision(robot, q)):
    #     success = False

    return q, success
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)
    
    
    
