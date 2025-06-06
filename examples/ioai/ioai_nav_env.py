#####################################################################################
# Copyright (c) 2023-2025 Galbot. All Rights Reserved.
#
# This software contains confidential and proprietary information of Galbot, Inc.
# ("Confidential Information"). You shall not disclose such Confidential Information
# and shall use it only in accordance with the terms of the license agreement you
# entered into with Galbot, Inc.
#
# UNAUTHORIZED COPYING, USE, OR DISTRIBUTION OF THIS SOFTWARE, OR ANY PORTION OR
# DERIVATIVE THEREOF, IS STRICTLY PROHIBITED. IF YOU HAVE RECEIVED THIS SOFTWARE IN
# ERROR, PLEASE NOTIFY GALBOT, INC. IMMEDIATELY AND DELETE IT FROM YOUR SYSTEM.
#####################################################################################
#          _____             _   _       _   _
#         / ____|           | | | |     | \ | |
#        | (___  _   _ _ __ | |_| |__   |  \| | _____   ____ _
#         \___ \| | | | '_ \| __| '_ \  | . ` |/ _ \ \ / / _` |
#         ____) | |_| | | | | |_| | | | | |\  | (_) \ V / (_| |
#        |_____/ \__, |_| |_|\__|_| |_| |_| \_|\___/ \_/ \__,_|
#                 __/ |
#                |___/
#
#####################################################################################
#
# Description: Navigation env setup using Galbot
# Author: Chenyu Cao@Galbot
# Date: 2025-05-24
#
#####################################################################################

from physics_simulator import PhysicsSimulator
from synthnova_config import PhysicsSimulatorConfig, RobotConfig, MujocoConfig
from physics_simulator.utils.data_types import JointTrajectory
from pathlib import Path
from physics_simulator.galbot_interface import (
    GalbotInterface,
    GalbotInterfaceConfig
)
from synthnova_config import CuboidConfig
import numpy as np
import random

from physics_simulator.utils.path_planner import AStarPathPlanner

def interpolate_joint_positions(start_positions, end_positions, steps):
    return np.linspace(start_positions, end_positions, steps).tolist()

class IoaiNavEnv:
    def __init__(self, headless=False):
        self.simulator = None
        self.robot = None
        self.interface = None

        self.planner = AStarPathPlanner(grid_size=1, obstacle_radius=0.5)
        self.start_pos = (0, 0)
        self.goal_pos = (10, 10)

        self._setup_simulator(headless)
        self._setup_interface()

        self._init_pose()

        self.path = self.planner.find_path(self.start_pos, self.goal_pos)
        self.waypoint_index = 0

    def _setup_simulator(self, headless):
        """
        Initialize the physics simulator with basic configuration.
        
        Args:
            headless: Whether to run in headless mode
        """
        # Create simulator config
        sim_config = PhysicsSimulatorConfig(
            mujoco_config=MujocoConfig(headless=headless)
        )
        
        # Initialize the simulator
        self.simulator = PhysicsSimulator(sim_config)
        
        # Add default scene
        self.simulator.add_default_scene()

        # Add robot
        robot_config = RobotConfig(
            prim_path="/World/Galbot",
            name="galbot_one_charlie",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("robot")
            .joinpath("galbot_one_charlie_description")
            .joinpath("galbot_one_charlie.xml"),
            position=[0, 0, 0],
            orientation=[0, 0, 0, 1]
        )
        self.simulator.add_robot(robot_config)

        # Initialize the scene
        self._init_scene()
        obstacles = self._add_random_obstables()
        
        # Initialize the simulator
        self.simulator.initialize()
        
        # Get robot instance for joint name discovery
        self.robot = self.simulator.get_robot("/World/Galbot")

    def _init_scene(self):
        """
        Initialize the scene with tables, closet, and cubes.
        """
        pass

    def _setup_interface(self):
        galbot_interface_config = GalbotInterfaceConfig()

        galbot_interface_config.robot.prim_path = "/World/Galbot"

        robot_name = self.robot.name
        # Enable modules
        galbot_interface_config.modules_manager.enabled_modules.append("right_arm")
        galbot_interface_config.modules_manager.enabled_modules.append("left_arm")
        galbot_interface_config.modules_manager.enabled_modules.append("leg")
        galbot_interface_config.modules_manager.enabled_modules.append("head")
        galbot_interface_config.modules_manager.enabled_modules.append("chassis")

        galbot_interface_config.right_arm.joint_names = [
            f"{robot_name}/right_arm_joint1",
            f"{robot_name}/right_arm_joint2",
            f"{robot_name}/right_arm_joint3",
            f"{robot_name}/right_arm_joint4",
            f"{robot_name}/right_arm_joint5",
            f"{robot_name}/right_arm_joint6",
            f"{robot_name}/right_arm_joint7",
        ]

        galbot_interface_config.left_arm.joint_names = [
            f"{robot_name}/left_arm_joint1",
            f"{robot_name}/left_arm_joint2",
            f"{robot_name}/left_arm_joint3",
            f"{robot_name}/left_arm_joint4",
            f"{robot_name}/left_arm_joint5",
            f"{robot_name}/left_arm_joint6",
            f"{robot_name}/left_arm_joint7",
        ]

        galbot_interface_config.leg.joint_names = [
            f"{robot_name}/leg_joint1",
            f"{robot_name}/leg_joint2",
            f"{robot_name}/leg_joint3",
            f"{robot_name}/leg_joint4",
        ]
        
        galbot_interface_config.head.joint_names = [
            f"{robot_name}/head_joint1",
            f"{robot_name}/head_joint2"
        ]

        galbot_interface_config.chassis.joint_names = [
            f"{robot_name}/mobile_forward_joint",
            f"{robot_name}/mobile_side_joint",
            f"{robot_name}/mobile_yaw_joint",
        ]

        galbot_interface = GalbotInterface(
            galbot_interface_config=galbot_interface_config,
            simulator=self.simulator
        )
        galbot_interface.initialize()

        self.interface = galbot_interface

    def _init_pose(self):
        # Init head pose
        head = [0.0, 0.0]
        self._move_joints_to_target(self.interface.head, head)

        # Init leg pose
        leg = [0.43, 1.48, 1.07, 0.0]
        self._move_joints_to_target(self.interface.leg, leg)

        # Init left arm pose
        left_arm = [
            0.058147381991147995,
            1.4785659313201904,
            -0.0999724417924881,
            -2.097979784011841,
            1.3999720811843872,
            -0.009971064515411854,
            1.0999830961227417,
        ]
        self._move_joints_to_target(self.interface.left_arm, left_arm)

        # Init right arm pose
        right_arm = [
            -0.058147381991147995,
            -1.4785659313201904,
            0.0999724417924881,
            2.097979784011841,
            -1.3999720811843872,
            0.009971064515411854,
            -1.0999830961227417,
        ]
        self._move_joints_to_target(self.interface.right_arm, right_arm)

    def _add_random_obstables(self):        
        obstacle_points = self.planner.generate_obstacles(probability=0.3, exclusion_zones=[(0, 0, 2), (10, 10, 2)])
        for i, point in enumerate(obstacle_points):
            
            self.simulator.add_object(
                CuboidConfig(
                    prim_path=f"/World/Obstacle_{i}",
                    name=f"obstacle_{i}",
                    position=[point[0], point[1], 0.4],
                    scale=[0.6, 0.6, 0.6],
                    color=[random.random(), random.random(), random.random()]
                )
            )

    def _move_joints_to_target(self, module, target_positions, steps=200):
        """Move joints from current position to target position smoothly."""
        current_positions = module.get_joint_positions()
        positions = interpolate_joint_positions(current_positions, target_positions, steps)
        joint_trajectory = JointTrajectory(positions=np.array(positions))
        module.follow_trajectory(joint_trajectory)
        

    def _is_joint_positions_reached(self, module, target_positions):
        current_positions = module.get_joint_positions()
        return np.allclose(current_positions, target_positions, atol=0.1)
    
    def follow_path_callback(self):
        if self.simulator.get_step_count() < 3000:
            return

        if self.waypoint_index < len(self.path) - 1:
            waypoint = self.path[self.waypoint_index]
            target_joint_positions = [waypoint[0], waypoint[1], 0]
            if self._is_joint_positions_reached(self.interface.chassis, target_joint_positions):
                self.waypoint_index += 1
                waypoint = self.path[self.waypoint_index]
                target_joint_positions = [waypoint[0], waypoint[1], 0]
                self._move_joints_to_target(self.interface.chassis, target_joint_positions)
        else:
            self.simulator.remove_physics_callback("follow_path_callback")

if __name__ == "__main__":
    env = IoaiNavEnv(headless=False)

    env.simulator.play()

    env.simulator.add_physics_callback("follow_path_callback", env.follow_path_callback)
    
    env.simulator.loop()

    env.simulator.close()