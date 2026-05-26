#!/usr/bin/env python3

import numpy as np
import math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

DEFAULT_REWARD_PARAMS = {
    # Collision handling
    'collision_threshold': 0.3,
    'collision_penalty': -100.0,
    # Speed profile
    'max_speed': 3.0,
    'min_speed': 0.0,
    'straight_speed': 3.0,
    'corner_speed': 1.2,
    'straight_distance': 6.0,
    'corner_distance': 2.0,
    'curvature_threshold': 0.2,
    'speed_limit_buffer': 0.2,
    'speed_sigma': 0.8,
    'speed_reward_weight': 1.0,
    'overspeed_tolerance': 0.2,
    'overspeed_penalty_weight': -1.5,
    # Progress
    'progress_reward_weight': 4.0,
    # Centerline tracking
    'centerline_reward_weight': 1.0,
    'centerline_sigma': 6.0,
    'centerline_change_penalty_weight': -0.3,
    # Steering smoothness
    'steering_change_penalty_weight': -0.4,
    'steering_accel_penalty_weight': -0.2,
    # Obstacle clearance
    'obstacle_distance_reward_weight': 0.2,
    'obstacle_distance_cap': 1.0,
    # Lidar processing
    'max_lidar_range': 10.0,
    'scan_window': 5
}

def _merge_params(config):
    params = DEFAULT_REWARD_PARAMS.copy()
    if config:
        params.update(config)
    return params

def _slice_window(values, center_idx, window):
    start = max(center_idx - window, 0)
    end = min(center_idx + window + 1, len(values))
    return values[start:end]

def analyze_scan(scan, params=None):
    params = _merge_params(params)
    ranges = np.array(scan.ranges, dtype=np.float32)
    max_range = params['max_lidar_range']
    ranges = np.where(np.isfinite(ranges), ranges, max_range)
    if ranges.size == 0:
        return {
            'min_distance': max_range,
            'front_distance': max_range,
            'left_distance': max_range,
            'right_distance': max_range,
            'centerline_error': 0.0,
            'curvature_proxy': 0.0
        }

    min_distance = float(np.min(ranges))
    center_idx = len(ranges) // 2
    left_idx = len(ranges) // 4
    right_idx = (3 * len(ranges)) // 4
    window = int(min(params['scan_window'], max(1, len(ranges) // 8)))

    front_slice = _slice_window(ranges, center_idx, window)
    left_slice = _slice_window(ranges, left_idx, window)
    right_slice = _slice_window(ranges, right_idx, window)

    front_distance = float(np.mean(front_slice)) if front_slice.size else max_range
    left_distance = float(np.mean(left_slice)) if left_slice.size else max_range
    right_distance = float(np.mean(right_slice)) if right_slice.size else max_range

    centerline_error = abs(left_distance - right_distance)
    curvature_proxy = centerline_error / max(left_distance + right_distance, 1e-3)

    return {
        'min_distance': min_distance,
        'front_distance': front_distance,
        'left_distance': left_distance,
        'right_distance': right_distance,
        'centerline_error': centerline_error,
        'curvature_proxy': curvature_proxy
    }

def calculate_target_speed(scan, params=None, scan_features=None):
    params = _merge_params(params)
    features = scan_features or analyze_scan(scan, params)

    front_distance = features['front_distance']
    curvature_proxy = features['curvature_proxy']

    straight_distance = params['straight_distance']
    corner_distance = params['corner_distance']
    straight_speed = params['straight_speed']
    corner_speed = params['corner_speed']

    if (front_distance >= straight_distance and
            curvature_proxy < params['curvature_threshold']):
        target_speed = straight_speed
    elif (front_distance <= corner_distance or
          curvature_proxy >= params['curvature_threshold']):
        target_speed = corner_speed
    else:
        ratio = (front_distance - corner_distance) / max(straight_distance - corner_distance, 1e-3)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        target_speed = corner_speed + ratio * (straight_speed - corner_speed)

    return float(np.clip(target_speed, params['min_speed'], params['max_speed']))

def calculate_reward(
    scan,
    odom,
    prev_odom,
    action=None,
    prev_action=None,
    prev_prev_action=None,
    prev_centerline_error=None,
    config=None
):
    """
    Calculate reward for the current state and action
    
    Args:
        scan: Current laser scan
        odom: Current odometry
        prev_odom: Previous odometry
        
    Returns:
        reward: Calculated reward value
        done: Whether the episode is done
    """
    params = _merge_params(config)
    reward = 0.0
    done = False
    
    # 1. Collision penalty
    scan_features = analyze_scan(scan, params)
    min_distance = scan_features['min_distance']
    if min_distance < params['collision_threshold']:
        return params['collision_penalty'], True
    
    # 2. Speed reward
    current_speed = _get_speed_from_odom(odom)
    target_speed = calculate_target_speed(scan, params, scan_features)
    speed_error = current_speed - target_speed
    speed_reward = params['speed_reward_weight'] * math.exp(
        -0.5 * (speed_error / params['speed_sigma'])**2
    )
    reward += speed_reward

    overspeed = max(0.0, current_speed - (target_speed + params['overspeed_tolerance']))
    reward += params['overspeed_penalty_weight'] * overspeed
    
    # 3. Distance to obstacles reward
    # Higher reward for keeping distance from obstacles
    distance_reward = params['obstacle_distance_reward_weight'] * min(
        min_distance, params['obstacle_distance_cap']
    ) / params['obstacle_distance_cap']
    reward += distance_reward
    
    # 4. Progress reward
    # Calculate distance traveled
    distance_traveled = 0.0
    current_pose = odom.pose.pose
    if prev_odom is not None:
        prev_pose = prev_odom.pose.pose

        dx = current_pose.position.x - prev_pose.position.x
        dy = current_pose.position.y - prev_pose.position.y
        distance_traveled = math.sqrt(dx**2 + dy**2)
    
    # Encourage forward movement
    progress_reward = params['progress_reward_weight'] * distance_traveled
    reward += progress_reward
    
    # 5. Steering efficiency reward
    # Penalize excessive steering
    steering_change = 0.0
    steering_accel = 0.0
    if action is not None and prev_action is not None:
        steering_change = abs(action[0] - prev_action[0])
        if prev_prev_action is not None:
            steering_accel = abs(action[0] - 2 * prev_action[0] + prev_prev_action[0])
    reward += params['steering_change_penalty_weight'] * steering_change
    reward += params['steering_accel_penalty_weight'] * steering_accel
    
    # 6. Centerline following reward
    # Perfect centerline following would have equal distances on both sides
    centerline_error = scan_features['centerline_error']
    centerline_reward = params['centerline_reward_weight'] * math.exp(
        -params['centerline_sigma'] * centerline_error
    )
    reward += centerline_reward

    if prev_centerline_error is not None:
        centerline_change = abs(centerline_error - prev_centerline_error)
        reward += params['centerline_change_penalty_weight'] * centerline_change
    
    return reward, done

def _get_speed_from_odom(odom):
    """Extract linear speed from odometry message"""
    vx = odom.twist.twist.linear.x
    vy = odom.twist.twist.linear.y
    return math.sqrt(vx**2 + vy**2)
