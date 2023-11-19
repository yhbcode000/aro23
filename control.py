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
from config import LEFT_HAND, RIGHT_HAND

from setup_pybullet import Simulation

from scipy.optimize import minimize
from config import OBSTACLE_PLACEMENT
from tools import distanceToObstacle  # Assuming Bezier class is defined as before



# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 200000.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

def segment_interpolation(path, number_seg):
    new_q_list = []
    for i in range(len(path) - 1):
        q_list = np.linspace(path[i], path[i+1], number_seg)
        if new_q_list == []:
            new_q_list = q_list
            continue
        new_q_list = np.concatenate((new_q_list, q_list))
    
    new_q_list = np.concatenate((new_q_list, np.array([path[-1]])))

    return new_q_list

def constant_velocity_segment_interpolation(robot, path, total_time, number_seg, v_max, a_max):
    dt = total_time / (number_seg * len(path))
    
    if v_max <= 0 or a_max <= 0 or dt <= 0:
        raise ValueError("Velocity, acceleration rate, and dt must be positive")

    new_q_list = []
    for i in range(len(path) - 1):
        if new_q_list == []:
            pq = path[i]
        else:
            pq = new_q_list[-1][0]
        segment_length = np.linalg.norm(np.array(np.array(path[i+1]) - pq))
        accel_distance = decel_distance = (v_max ** 2) / (2 * a_max)

        if 2 * accel_distance > segment_length:
            # Adjust V and distances if segment is too short for full acceleration and deceleration
            accel_distance = decel_distance = segment_length / 2
            v_max = np.sqrt(a_max * segment_length)

        const_velocity_distance = segment_length - (accel_distance + decel_distance)
        total_segment_time = 2 * (v_max / a_max) + const_velocity_distance / v_max

        # Check if total_segment_time is a valid number
        if not np.isfinite(total_segment_time):
            raise ValueError("Total segment time is not finite. Check your input values.")

        current_position = 0
        current_velocity = 0
        for t in np.arange(0, total_segment_time, dt):
            if t < v_max / a_max:  # Acceleration phase
                current_velocity += a_max * dt
                a = a_max
            elif t < total_segment_time - (v_max / a_max):  # Constant velocity
                current_velocity = v_max
                a = 0
            else:  # Deceleration phase
                current_velocity -= a_max * dt
                a = a_max

            # Update position and interpolate
            current_position += current_velocity * dt
            ratio = current_position / segment_length
            new_q = (1 - ratio) * np.array(path[i]) + ratio * np.array(path[i+1])
            new_q_list.append(new_q)

    return np.array(new_q_list)

def cost_function(control_points_index, construct_points, total_time, robot, v_max, a_max, min_distance):
    # Convert control points to a Bezier curve    
    q_of_t = Bezier(construct_points(control_points_index), t_min=0, t_max=total_time)
    vq_of_t = q_of_t.derivative(1)
    vvq_of_t = vq_of_t.derivative(1)
    
    # Sample points on the Bezier curve and evaluate the cost based on robot dynamics
    t_samples = np.linspace(0, total_time, 100)
    total_cost = 0
    for i, t in enumerate(t_samples):
        point = q_of_t(t)
        velocity = vq_of_t(t)
        acceleration = vvq_of_t(t)
        
        if np.any(velocity > v_max) or np.any(velocity < - v_max):
            velocity_cost = abs((velocity - v_max) / v_max)
            total_cost += np.linalg.norm(velocity_cost)
            
        if np.any(acceleration > a_max) or np.any(acceleration < - a_max):
            acceleration_cost = abs((acceleration - a_max) / a_max)
            total_cost += np.linalg.norm(acceleration_cost)

        if i == 0 or i == len(t_samples) - 1:
            v_min = 1e-4
            if np.any(velocity > v_min) or np.any(velocity < - v_min):
                velocity_cost = abs(velocity)
                total_cost += np.linalg.norm(velocity_cost)
        
        # if distanceToObstacle(robot, point) < min_distance:
        #     collision_cost = min_distance - distanceToObstacle(robot, point)
        #     total_cost += collision_cost

    # print(f"total_cost: {total_cost}")

    return total_cost

def optimize_bezier_control_points(robot, initial_control_points, total_time = 1, v_max = 60, a_max = 20):

    # min_distance = min([distanceToObstacle(robot, q) for q in initial_control_points])
    
    def construct_points(control_points):
        return np.concatenate((initial_control_points[0:1],  
                              control_points.reshape(initial_control_points[1:-1].shape),
                              initial_control_points[-1:]), axis=0)
    
    # Flatten the control points array for optimization
    initial_guess = np.array(initial_control_points[1:-1]).flatten()

    # Optimize control points
    result = minimize(cost_function, initial_guess, args=(construct_points, total_time, robot, v_max, a_max, min_distance), method='SLSQP')
    
    # Reshape the optimized control points to their original shape    
    optimized_control_points = construct_points(result.x)
    return optimized_control_points

def filter_bezier_control_points(robot, initial_control_points, total_time = 1, number_sample_p_t = 10, v_max = 60, a_max = 20):
    number_keep = int(total_time * number_sample_p_t)
    
    if number_keep < 10:
        number_keep = 10
    
    min_distance = min([distanceToObstacle(robot, q) for q in initial_control_points])
    
    def construct_points(control_points_index):
        # control_points_index = sorted(control_points_index)
        points = initial_control_points[0:1]
        for i in control_points_index:
            index = min(int(i * len(initial_control_points)), len(initial_control_points) - 1)
            points = np.append(points, np.array([initial_control_points[index]]), axis=0)
        points = np.append(points, np.array([initial_control_points[-1]]), axis=0)
        # print(points.shape)
        return points

    initial_guess = np.linspace(0, 1, number_keep)
    
    # Optimize control points
    # result = minimize(cost_function, initial_guess, args=(construct_points, v_max, a_max, min_distance), method='SLSQP')
    result = minimize(cost_function, initial_guess, method='BFGS', args=(construct_points, total_time, robot, v_max, a_max, min_distance), bounds=(
        (0, 1) for _ in initial_guess
    ))
    
    # Reshape the optimized control points to their original shape  
    points = construct_points(result.x)
    last_cost = cost_function(result.x, construct_points, total_time, robot, v_max, a_max, min_distance)
    if last_cost != 0:
        print(f"LOG: Can not find the optimal solution, try with more sample points. \nCurrent sample points: {number_sample_p_t}; current cost: {last_cost}")
        return filter_bezier_control_points(robot, initial_control_points, total_time, number_sample_p_t + 5, v_max, a_max)
    else:
        return points
    
def maketraj(robot, path, total_time, number_sample_p_t=10, v_max = 360, a_max = 360): 
    path = segment_interpolation(path, number_seg=5)
    path = filter_bezier_control_points(robot, path, total_time, number_sample_p_t, v_max = v_max, a_max = a_max)

    q_of_t = Bezier(path,t_max=total_time)
    vq_of_t = q_of_t.derivative(1)
    vvq_of_t = vq_of_t.derivative(1)
    return q_of_t, vq_of_t, vvq_of_t

def controllaw(sim, robot, trajs, tcurrent, cube, viz = None):
    # Get the current state from the simulator
    q_c, vq_c = sim.getpybulletstate()
    q_of_t, vq_of_t, vvq_of_t = trajs

    # Desired states
    q_des = q_of_t(tcurrent)  # Desired position
    vq_des = vq_of_t(tcurrent)  # Desired velocity
    vvq_des = vvq_of_t(tcurrent)  # Desired acceleration # TODO add some force to the clamp

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
        
    # Get frame IDs
    left_frame_id = robot.model.getFrameId(LEFT_HAND)
    right_frame_id = robot.model.getFrameId(RIGHT_HAND)
    
    # Assuming force is a 6-dimensional vector
    force = np.array([0, -500 * np.linalg.norm(vq_c), 0, 0, 0, 0])

    # Compute Jacobians for the hands
    JL = pin.computeFrameJacobian(robot.model, robot.data, q_c, left_frame_id)
    JR = pin.computeFrameJacobian(robot.model, robot.data, q_c, right_frame_id)

    # Calculate torques from forces
    left_hand_torques = JL.T @ force
    right_hand_torques = JR.T @ force

    # Apply torques to the respective joints
    # Ensure torques array can accommodate these indices and is initialized properly
    torques += left_hand_torques
    torques += right_hand_torques
    
    # Update visuals and simulate
    sim.step(torques)
    
    return torques

if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    robot, sim, cube = setupwithpybullet()
    
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    
    
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)    
    
    total_time=4.
    trajs = maketraj(robot, path, total_time)   
    
    #setting initial configuration
    sim.setqsim(q0)
    
    tcur = 0.
    
    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT
    