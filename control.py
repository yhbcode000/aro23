#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import time
import numpy as np

from bezier import Bezier
import pinocchio as pin

from setup_meshcat import updatevisuals
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)


def constant_velocity_segment_interpolation(robot, path, V, acceleration_rate, dt):
    if V <= 0 or acceleration_rate <= 0 or dt <= 0:
        raise ValueError("Velocity, acceleration rate, and dt must be positive")

    new_q_list = []
    for i in range(len(path) - 1):
        if new_q_list == []:
            pq = path[i]
        else:
            pq = new_q_list[-1][0]
        segment_length = np.linalg.norm(np.array(np.array(path[i+1]) - pq))
        accel_distance = decel_distance = (V ** 2) / (2 * acceleration_rate)

        if 2 * accel_distance > segment_length:
            # Adjust V and distances if segment is too short for full acceleration and deceleration
            accel_distance = decel_distance = segment_length / 2
            V = np.sqrt(acceleration_rate * segment_length)

        const_velocity_distance = segment_length - (accel_distance + decel_distance)
        total_segment_time = 2 * (V / acceleration_rate) + const_velocity_distance / V

        # Check if total_segment_time is a valid number
        if not np.isfinite(total_segment_time):
            raise ValueError("Total segment time is not finite. Check your input values.")

        current_position = 0
        current_velocity = 0
        for t in np.arange(0, total_segment_time, dt):
            if t < V / acceleration_rate:  # Acceleration phase
                current_velocity += acceleration_rate * dt
                a = acceleration_rate
            elif t < total_segment_time - (V / acceleration_rate):  # Constant velocity
                current_velocity = V
                a = 0
            else:  # Deceleration phase
                current_velocity -= acceleration_rate * dt
                a = acceleration_rate

            # Update position and interpolate
            current_position += current_velocity * dt
            ratio = current_position / segment_length
            new_q = (1 - ratio) * np.array(path[i]) + ratio * np.array(path[i+1])
            new_q_list.append((new_q, current_velocity, a))

    return new_q_list

def get_v_a(t, path_with_dynamics, dt):
    index = int(t//dt)
    if index >= len(path_with_dynamics):
        return path_with_dynamics[-1]
    return path_with_dynamics[index]


def display_path_with_dynamics(path_with_dynamics, dt, viz):
    for q, v, a in path_with_dynamics:
        viz.display(q)
        time.sleep(dt)

def calculate_total_distance(path):
    """
    Calculate the total Euclidean distance covered in the path.
    """
    total_distance = 0
    for i in range(1, len(path)):
        total_distance += np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
    return total_distance


def interpolate_position(path, position):
    """
    Interpolate the robot's configuration based on the position along the path.
    """
    accumulated_distance = 0
    for i in range(1, len(path)):
        segment_length = np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
        if accumulated_distance + segment_length >= position:
            ratio = (position - accumulated_distance) / segment_length
            return (1 - ratio) * np.array(path[i-1]) + ratio * np.array(path[i])
        accumulated_distance += segment_length
    return path[-1]

def maketraj(robot, path, T, dt=0.01): 
    path_with_dynamics = constant_velocity_segment_interpolation(robot, path,
                                                V = 60,
                                                acceleration_rate = 20, 
                                                dt=dt) # TODO this function will deprecate soon
        
    path = [q for q, _, _ in path_with_dynamics]
        
    q_of_t = Bezier(path,t_max=T)
    vq_of_t = q_of_t.derivative(1)
    vvq_of_t = vq_of_t.derivative(1)
    return q_of_t, vq_of_t, vvq_of_t

def controllaw(sim, robot, trajs, tcurrent, cube, viz=None):
    # Get the current state from the simulator
    q_c, vq_c = sim.getpybulletstate()
    q_of_t, vq_of_t, vvq_of_t = trajs

    # Desired states
    q_des = q_of_t(tcurrent)  # Desired position
    vq_des = vq_of_t(tcurrent)  # Desired velocity
    vvq_des = vvq_of_t(tcurrent)  # Desired acceleration

    # Position and velocity errors
    pos_err = q_des - q_c
    vel_err = vq_des - vq_c

    # PD control for position and velocity
    vvq = Kp * pos_err + Kv * vel_err + vvq_des

    # Update robot's data structure for the current state
    pin.forwardKinematics(robot.model, robot.data, q_c, vq_c)
    pin.computeJointJacobians(robot.model, robot.data, q_c)

    # Compute the Mass matrix using CRBA
    M = pin.crba(robot.model, robot.data, q_c)

    # Compute Coriolis forces (not matrix)
    coriolis_forces = pin.nonLinearEffects(robot.model, robot.data, q_c, vq_c)

    # Compute gravitational torques
    G = pin.computeGeneralizedGravity(robot.model, robot.data, q_c)

    # Calculate torques
    torques = M.dot(vvq) + coriolis_forces + G

    # Update visuals and simulate
    updatevisuals(viz, robot, cube, q_des) if viz else None
    sim.step(torques)

if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    robot, sim, cube = setupwithpybullet()
    
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(robot, q0, qe, cube, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    path_with_dynamics = constant_velocity_segment_interpolation(robot, path,
                                                V = 60,
                                                acceleration_rate = 20, 
                                                dt=DT)
    
    #setting initial configuration
    sim.setqsim(q0)
    
    
    #TODO this is just an example, you are free to do as you please.
    #In any case this trajectory does not follow the path 
    #0 init and end velocities
    def maketraj(path, T): #TODO compute a real trajectory !
        q_of_t = Bezier(path,t_max=T)
        vq_of_t = q_of_t.derivative(1)
        vvq_of_t = vq_of_t.derivative(1)
        return q_of_t, vq_of_t, vvq_of_t


    #TODO this is just a random trajectory, you need to do this yourself
    total_time=1.
    trajs = maketraj([q for q, _, _ in path_with_dynamics], total_time)   

    tcur = 0.
    
    
    

    
    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT
        
    
    