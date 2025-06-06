#####################################################################################
# Copyright (c) 2023-2025 Galbot. All Rights Reserved.
# (Licence header unchanged – omitted here for brevity)
#####################################################################################
# ioai_direct_move_env.py  – continuous glide to an absolute target
# -----------------------------------------------------------------
from physics_simulator import PhysicsSimulator
from synthnova_config import PhysicsSimulatorConfig, RobotConfig, MujocoConfig
from physics_simulator.utils.data_types import JointTrajectory
from pathlib import Path
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
import numpy as np
import math

# ————————————————————————————————————————————————————————————
def _interp(start, end, steps):
    return np.linspace(start, end, steps).tolist()

class IoaiDirectMoveEnv:
    def __init__(self, goal=(0.0, 5.0), duration=1.0, headless=False):
        self.goal = goal
        self.duration = duration
        self.t0 = None                      # wall-clock offset

        self.simulator = None
        self.interface = None
        self._setup_sim(headless)
        self._setup_interface()
        self._set_initial_pose()

        self.simulator.add_physics_callback("glide_to_goal", self._glide_cb)

    # ----- simulator + robot -----
    def _setup_sim(self, headless):
        cfg = PhysicsSimulatorConfig(mujoco_config=MujocoConfig(headless=headless))
        self.simulator = PhysicsSimulator(cfg)
        self.simulator.add_default_scene()

        xml = (Path(self.simulator.synthnova_assets_directory)
               / "synthnova_assets/robot/galbot_one_charlie_description/galbot_one_charlie.xml")
        rob_cfg = RobotConfig(prim_path="/World/Galbot", name="galbot_one_charlie",
                              mjcf_path=xml, position=[0,0,0], orientation=[0,0,0,1])
        self.simulator.add_robot(rob_cfg)
        self.simulator.initialize()

    def _setup_interface(self):
        cfg = GalbotInterfaceConfig()
        cfg.robot.prim_path = "/World/Galbot"
        name = "galbot_one_charlie"
        cfg.modules_manager.enabled_modules += ["right_arm","left_arm","leg","head","chassis"]
        cfg.right_arm.joint_names = [f"{name}/right_arm_joint{i}" for i in range(1,8)]
        cfg.left_arm.joint_names  = [f"{name}/left_arm_joint{i}"  for i in range(1,8)]
        cfg.leg.joint_names       = [f"{name}/leg_joint{i}"       for i in range(1,5)]
        cfg.head.joint_names      = [f"{name}/head_joint1", f"{name}/head_joint2"]
        cfg.chassis.joint_names   = [f"{name}/mobile_forward_joint",
                                     f"{name}/mobile_side_joint",
                                     f"{name}/mobile_yaw_joint"]
        self.interface = GalbotInterface(cfg, simulator=self.simulator)
        self.interface.initialize()

    def _set_initial_pose(self):
        head = [0,0]
        leg  = [0.43, 1.48, 1.07, 0]
        arm  = [0.06, 1.48, -0.1, -2.1, 1.4, -0.01, 1.1]
        traj_sets = [
            (self.interface.head, head),
            (self.interface.leg, leg),
            (self.interface.left_arm,  arm),
            (self.interface.right_arm, [-x for x in arm]),
        ]
        for mod,target in traj_sets:
            traj = JointTrajectory(positions=np.array(_interp(mod.get_joint_positions(), target, 200)))
            mod.follow_trajectory(traj)

    # ----- continuous glide callback -----
    def _glide_cb(self):
        step = self.simulator.get_step_count()
        if step < 50:                      # let sensors & IK settle
            return
        if self.t0 is None:
            self.t0 = self.simulator.get_simulation_time()

        t_now = self.simulator.get_simulation_time() - self.t0
        if t_now >= self.duration:
            # final snap to goal + remove callback
            self.interface.chassis.set_joint_positions([*self.goal, 0.0])
            self.simulator.remove_physics_callback("glide_to_goal")
            return

        # linear time-based interpolation
        alpha = t_now / self.duration
        cur   = self.interface.chassis.get_joint_positions()
        x_des = cur[0] + alpha * (self.goal[0] - cur[0])
        y_des = cur[1] + alpha * (self.goal[1] - cur[1])
        self.interface.chassis.set_joint_positions([x_des, y_des, 0.0])

    # ----- run helper -----
    def run(self):
        try:
            self.simulator.play()
            self.simulator.loop()
        finally:
            self.simulator.close()

# quick test
if __name__ == "__main__":
    IoaiDirectMoveEnv(goal=(0,5), duration=2.0, headless=False).run()
