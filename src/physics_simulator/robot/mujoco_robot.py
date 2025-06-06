#####################################################################################
#
# Description: Mujoco simulator wrapper for SynthNova physics simulator
# Date: 2025-04-27
# Author: Chenyu Cao@Galbot
#
#####################################################################################

import traceback

import time
import os

import gc
from typing import List
from auro_utils import xyzw_to_wxyz
from auro_utils import Logger
from threading import RLock

import pathlib

from physics_simulator.simulator.base_sim import BaseSim
from synthnova_config import (
    PhysicsSimulatorConfig,
    LoggerConfig,
    MujocoConfig,
    RobotConfig,
    ObjectConfig,
    GroundPlaneConfig,
    SensorConfig,
    RgbCameraConfig,
    DepthCameraConfig,
    ScenarioConfig
)

from pathlib import Path
from auro_utils import wxyz_to_xyzw

from physics_simulator.utils.path_manager import PathManager

class MujocoSimulator(BaseSim):
    def __init__(self, physics_simulator_config: PhysicsSimulatorConfig):
        """Initialize the MuJoCo-based physics simulator.

        Args:
            physics_simulator_config: Configuration object containing all simulator settings
        """
        from mujoco import viewer

        super().__init__(physics_simulator_config)
        self.root_directory = PathManager.get_root_path()
        self.synthnova_assets_directory = self.get_synthnova_assets_directory()
        self.root_prim_path = "/World"
        self.config = physics_simulator_config
        self.logger = None
        self._running = False
        self.model = None
        self.data = None
        self.viewer = None
        self.world = None
        self._robots = {}
        self._sensors = {}
        self._objects = {}
        self._ground_planes = {}
        self._physics_callbacks = {}
        self.lock = RLock()
        self._render_context_offscreen = None

        # Add signal handler for graceful shutdown
        import signal

        signal.signal(signal.SIGINT, self._signal_handler)

        try:
            # Load logger
            self.logger = self._load_logger(physics_simulator_config.logger_config)
            # Load simulator
            self.world = self._load_simulator(physics_simulator_config.mujoco_config)
            # Load scenario
            if self.config.scenario_config is not None and (
                self.config.scenario_config.robots
                or self.config.scenario_config.objects
                or self.config.scenario_config.sensors
                or self.config.scenario_config.ground_planes
            ):
                self.import_scenario(self.config.scenario_config)
        except Exception as e:
            print(f"Physics Simulator initialization failed: {e}")
            raise e

        else:
            self.logger.log_success("Physics Simulator (MuJoCo) initialized.")

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C signal by gracefully shutting down the simulator and releasing resources."""
        self.logger.log_info("Received shutdown signal. Cleaning up...")
        # Set running flag to False and call close directly
        self._running = False
        try:
            self.close()
        except Exception as e:
            self.logger.log_error(f"Error during signal handler cleanup: {e}")
            # Force exit if cleanup fails
            import sys
            sys.exit(1)

    def load_model_from_xml_string(self, xml):
        """Load a MuJoCo model from an XML string.
        
        Args:
            xml: String containing MuJoCo model XML
        """
        import mujoco

        # from physics_simulator.simulator import MjModel, MjData
        self.model = mujoco.MjModel.from_xml_string(xml)

    def load_model_from_xml_file(self, xml_file):
        """Load a MuJoCo model from an XML file.
        
        Args:
            xml_file: Path to the XML file
        """
        import mujoco

        # from physics_simulator.simulator import MjModel, MjData
        self.model = mujoco.MjModel.from_xml_file(xml_file)

    def initialize(self):
        """Initialize the simulator by setting up the MuJoCo model, data structures, and viewer.
        Must be called before stepping the simulation."""
        import mujoco
        from physics_simulator.simulator.mj_wrapper import (
            MjRenderContextOffscreen,
            MjModel,
            MjData
        )

        with self.lock:
            # Load the world
            # Initialize the MjModel and MjData
            mujoco_model = self.world.get_model(mode="mujoco")
            self.model = MjModel(mujoco_model)
            self.data = MjData(self.model)
            for robot in self._robots.values():
                robot["instance"].initialize()

            headless = self.config.mujoco_config.headless

            if headless:
                self.viewer = None
            else:
                if self.viewer is None:
                    self.viewer = mujoco.viewer.launch_passive(
                        self.model._model, self.data._data
                    )
                
                self.viewer.opt.geomgroup[0] = False  # collision
                self.viewer.opt.geomgroup[1] = True   # visual
                self.viewer.opt.geomgroup[2] = True   # visual
                self.viewer.opt.geomgroup[3] = False  # collision

            self._mujoco_timestep = self.model._model.opt.timestep

            with self.lock:
                try:
                    # Create render context with proper error handling
                    render_context = MjRenderContextOffscreen(self, device_id=-1)
                    if render_context is None:
                        raise RuntimeError("Failed to create render context")
                    
                    # Add render context with proper cleanup
                    self.add_render_context(render_context)
                except Exception as e:
                    self.logger.log_error(f"Error initializing render context: {e}")
                    raise

            self._running = True

    def forward(self):
        """Run forward dynamics to synchronize derived quantities in MuJoCo.
        This updates all computed quantities in the simulation without advancing time."""
        import mujoco

        with self.lock:
            mujoco.mj_forward(self.model._model, self.data._data)

    def step(self, num_steps: int = 1, render: bool = True):
        """Step simulation and execute physics callbacks."""
        import mujoco

        if self._running:
            if not render:
                self.logger.log_warning("Render parameter is ignored in Mujoco")
            for _ in range(num_steps):
                mujoco.mj_step(self.model._model, self.data._data)

            if self.viewer is not None:
                self.viewer.sync()

        else:
            raise RuntimeError(
                "Simulator is not running, please call initialize() first"
            )

    def step1(self):
        """Execute the first part of a simulation step (before control inputs).
        This computes forward dynamics without integrating positions."""
        import mujoco

        with self.lock:
            mujoco.mj_step1(self.model._model, self.data._data)

    def step2(self):
        """Execute the second part of a simulation step (after control inputs).
        This integrates positions and updates the simulation state."""
        import mujoco

        with self.lock:
            mujoco.mj_step2(self.model._model, self.data._data)

    def render(
        self,
        width=None,
        height=None,
        *,
        camera_name=None,
        depth=False,
        mode="offscreen",
        device_id=-1,
        segmentation=False,
    ):
        """Render the current scene from a specific camera view.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            camera_name: Name of the camera to render from
            depth: Whether to render depth information
            mode: Rendering mode ('offscreen' or 'window')
            device_id: GPU device ID to use for rendering
            segmentation: Whether to render segmentation mask
            
        Returns:
            numpy.ndarray: The rendered image
        """
        if camera_name is None:
            camera_id = None
        else:
            camera_id = self.model.camera_name2id(camera_name)

        assert mode == "offscreen", "only offscreen supported for now"
        assert self._render_context_offscreen is not None
        with self.lock:
            self._render_context_offscreen.render(
                width=width,
                height=height,
                camera_id=camera_id,
                segmentation=segmentation,
            )
            return self._render_context_offscreen.read_pixels(
                width, height, depth=depth, segmentation=segmentation
            )

    def add_render_context(self, render_context):
        assert render_context.offscreen
        if self._render_context_offscreen is not None:
            # free context
            del self._render_context_offscreen
        self._render_context_offscreen = render_context

    def is_running(self) -> bool:
        """Check if the simulator is initialized and running.
        
        Returns:
            bool: True if the simulator is running, False otherwise
        """
        return self._running

    def play(self):
        """Start the simulation if it's not already running."""
        if not self._running:
            self.initialize()

    def loop(self):
        """Run the simulation loop."""
        last_time = time.time()
        while self._running:
            current_time = time.time()
            elapsed = current_time - last_time
            
            num_steps = int(elapsed / self.config.mujoco_config.timestep)

            self.step(num_steps)

            if self._physics_callbacks:
                callbacks = list(self._physics_callbacks.values())
                for callback in callbacks:
                    callback()

            last_time = current_time

            time.sleep(self.config.mujoco_config.timestep)
        self.close()

    def reset(self):
        """Reset the simulation to its initial state."""
        import mujoco

        with self.lock:
            mujoco.mj_resetData(self.model, self.data)

    def close(self):
        """Clean up and close the simulator, releasing all resources."""
        with self.lock:
            # Set flag first to prevent any new operations
            self._running = False
            
            if self._render_context_offscreen is not None:
                del self._render_context_offscreen
                self._render_context_offscreen = None
            
            if self.viewer is not None:
                self.viewer.close()
            
            self.free()

    def free(self):
        """Clean up resources and prevent memory leaks."""
        with self.lock:
            if hasattr(self, 'data') and self.data is not None:
                del self.data
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            gc.collect()

    def add_default_scene(self, ground_plane_config: GroundPlaneConfig=None):
        """Add a default scene with ground plane to the simulation.
        
        Args:
            ground_plane_config: Optional configuration for the ground plane
        """
        from physics_simulator.object import MujocoXML

        name = "default_ground_plane"
        ground_plane_prim_path = (
            Path()
            .joinpath(self.root_prim_path)
            .joinpath(name)
        ).as_posix()

        ground_plane_xml_path = (
            Path()
            .joinpath(self.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("default_assets")
            .joinpath("default_ground_plane")
            .joinpath(f"default_ground_plane.xml")
        ).as_posix()

        if not Path(ground_plane_xml_path).exists():
            self.logger.log_error(f"Ground plane XML file not found: {ground_plane_xml_path}")
            raise FileNotFoundError(f"Ground plane XML file not found: {ground_plane_xml_path}")

        default_scene = MujocoXML(ground_plane_xml_path)

        ground_plane_config = GroundPlaneConfig()
        
        self._ground_planes[ground_plane_prim_path] = {
            "config": ground_plane_config,
            "instance": default_scene,
            "initialized": True,
            "metadata": {},
        }
        self.world.merge(default_scene)

        return ground_plane_prim_path

    def add_robot(self, robot_config: RobotConfig):
        """Add a robot to the simulation environment.
        
        Args:
            robot_config: Configuration for the robot to add
            
        Returns:
            str: Path to the added robot
        """
        from physics_simulator.robot import MujocoRobot

        # Use robot name as namespace
        namespace = robot_config.name + '/' if robot_config.name else 'robot/'
        
        # Create robot with namespace
        robot = MujocoRobot(self, robot_config, namespace=namespace)
        self.world.merge(robot.robot_model)

        # Store both mappings: prim_path -> robot and robot -> robot
        prim_path = robot.prim_path
        self._robots[prim_path] = {
            "config": robot_config,
            "instance": robot,
            "initialized": True,
            "metadata": {},
        }
        self.logger.log_debug(f"Added robot with prim_path: {prim_path}")

        return robot.prim_path

    def get_robot(self, prim_path):
        """Retrieve a robot by its path.

        Args:
            prim_path: Path identifier for the robot

        Returns:
            Robot instance
        """
        return self._robots[prim_path]["instance"]

    def remove_robot(self, prim_path: str):
        """Remove a robot from the simulator.

        Args:
            prim_path: Path identifier for the robot
        """
        self._robots.pop(prim_path)
        # TODO@Chenyu Cao: remove robot from world, and log the removal

    def add_object(self, object_config: ObjectConfig):
        """Add a physical object to the simulation environment.
        
        Args:
            object_config: Configuration for the object to add
            
        Returns:
            str: Path to the added object
        """
        from physics_simulator.object import MujocoObjectFactory

        # Create object using factory
        try:
            # xyzw to wxyz
            object_config_copy = object_config.model_copy()
            if object_config_copy.orientation is not None:
                object_config_copy.orientation = xyzw_to_wxyz(object_config_copy.orientation)
            if object_config_copy.rotation is not None:
                object_config_copy.rotation = xyzw_to_wxyz(object_config_copy.rotation)
            obj = MujocoObjectFactory.create_from_config(object_config_copy)
            self.world.merge_objects([obj])

            # Store object reference if using prim_path
            if hasattr(object_config, "prim_path") and object_config.prim_path:
                prim_path = object_config.prim_path
                self._objects[prim_path] = {
                    "config": object_config,
                    "instance": obj,
                    "initialized": True,
                    "metadata": {},
                }
                self.logger.log_debug(f"Added object with prim_path: {prim_path}")

            return object_config_copy.prim_path
        except ValueError as e:
            raise ValueError(f"Failed to add object: {e}")

    def get_object(self, prim_path: str):
        """Retrieve an object by its path.

        Args:
            prim_path: Path identifier for the object

        Returns:
            Object instance
        """
        return self._objects[prim_path]["instance"]

    def remove_object(self, prim_path: str):
        """Remove an object from the simulator.

        Args:
            prim_path: Path identifier for the object
        """
        obj = self._objects[prim_path]["instance"]
        self._objects.pop(prim_path)
        self._objects.pop(obj)

    def add_sensor(self, sensor_config: SensorConfig):
        """Add a sensor to the simulation environment.
        
        Args:
            sensor_config: Configuration for the sensor to add
            
        Returns:
            str: Path to the added sensor
        """
        self.logger.log_debug("Adding sensor...")
        if sensor_config.type == "rgb_camera":
            sensor = self._add_rgb_camera(sensor_config)
        elif sensor_config.type == "depth_camera":
            sensor = self._add_depth_camera(sensor_config)
        else:
            raise ValueError(f"Unsupported sensor type: {sensor_config.type}")

        prim_path = sensor_config.prim_path

        self._sensors[prim_path] = {
            "config": sensor_config,
            "instance": sensor,
            "initialized": True,
            "metadata": {},
        }
        return prim_path

    def _add_rgb_camera(self, sensor_config: RgbCameraConfig):
        from physics_simulator.sensor.rgb_camera import MujocoRgbCamera

        # Create RGB camera
        sensor = MujocoRgbCamera(simulator=self, camera_config=sensor_config)

        return sensor
    
    def _add_depth_camera(self, sensor_config: DepthCameraConfig):
        from physics_simulator.sensor.depth_camera import MujocoDepthCamera

        sensor = MujocoDepthCamera(simulator=self, camera_config=sensor_config)
        return sensor

    def get_sensor(self, prim_path: str):
        """Retrieve a sensor by its path.

        Args:
            prim_path: Path identifier for the sensor

        Returns:
            Sensor object
        """
        return self._sensors[prim_path]["instance"]

    def remove_sensor(self, prim_path: str):
        """Remove a sensor from the simulator.

        Args:
            prim_path: Path identifier for the sensor
        """
        self._sensors.pop(prim_path)

    def get_sensor_state(self, prim_path: str):
        """Get the current output/reading from a sensor.
        
        Args:
            prim_path: Path to the sensor
            
        Returns:
            Various: Sensor data (image for cameras, depth for depth sensors, etc.)
            
        Raises:
            KeyError: If sensor does not exist at the specified path
        """
        raise NotImplementedError("Sensor state is not implemented yet")

    def _load_logger(self, logger_config: LoggerConfig) -> Logger:
        """Load and configure the logging system.

        Args:
            logger_config: Logger configuration object

        Returns:
            Logger: Configured logger instance
        """
        logger = Logger(
            log_level=logger_config.log_level,
            use_file_log=logger_config.use_file_log,
            log_path=logger_config.log_path
            or os.path.join(self.root_directory, "logs"),
        )
        logger.log_info(f"Logger loaded with config: {logger_config.model_dump()}")
        return logger

    def _load_simulator(self, mujoco_config: MujocoConfig):
        """Load the simulator from the sim config.

        Args:
            mujoco_config: Simulator configuration object

        Returns:
            Simulator: Configured simulator instance
        """
        from physics_simulator.world import MujocoWorld

        world = MujocoWorld()
        # Update physics parameters
        if mujoco_config.timestep is not None:
            world.update_attribute("timestep", mujoco_config.timestep, tag="option")
        if mujoco_config.gravity is not None:
            world.update_attribute("gravity", mujoco_config.gravity, tag="option")
        if mujoco_config.density is not None:
            world.update_attribute("density", mujoco_config.density, tag="option")
        if mujoco_config.impratio is not None:
            world.update_attribute("impratio", mujoco_config.impratio, tag="option")

        # Update solver parameters
        if mujoco_config.viscosity is not None:
            world.update_attribute("viscosity", mujoco_config.viscosity, tag="option")
        if mujoco_config.integrator is not None:
            world.update_attribute("integrator", mujoco_config.integrator, tag="option")
        if mujoco_config.cone is not None:
            world.update_attribute("cone", mujoco_config.cone, tag="option")
        if mujoco_config.jacobian is not None:
            world.update_attribute("jacobian", mujoco_config.jacobian, tag="option")
        if mujoco_config.solver is not None:
            world.update_attribute("solver", mujoco_config.solver, tag="option")
        if mujoco_config.iterations is not None:
            world.update_attribute("iterations", mujoco_config.iterations, tag="option")
        if mujoco_config.tolerance is not None:
            world.update_attribute("tolerance", mujoco_config.tolerance, tag="option")

        # Update size settings
        if mujoco_config.size_config.memory is not None:
            world.update_attribute(
                "memory", mujoco_config.size_config.memory, tag="size"
            )

        # Update compiler settings
        if mujoco_config.compiler_config.strippath is not None:
            world.update_attribute(
                "strippath", mujoco_config.compiler_config.strippath, tag="compiler"
            )
        if mujoco_config.compiler_config.meshdir is not None:
            world.update_attribute(
                "meshdir", mujoco_config.compiler_config.meshdir, tag="compiler"
            )
        if mujoco_config.compiler_config.angle is not None:
            world.update_attribute(
                "angle", mujoco_config.compiler_config.angle, tag="compiler"
            )
        if mujoco_config.compiler_config.autolimits is not None:
            world.update_attribute(
                "autolimits", mujoco_config.compiler_config.autolimits, tag="compiler"
            )
        if mujoco_config.compiler_config.boundmass is not None:
            world.update_attribute(
                "boundmass", mujoco_config.compiler_config.boundmass, tag="compiler"
            )
        if mujoco_config.compiler_config.boundinertia is not None:
            world.update_attribute(
                "boundinertia",
                mujoco_config.compiler_config.boundinertia,
                tag="compiler",
            )
        if mujoco_config.compiler_config.discardvisual is not None:
            world.update_attribute(
                "discardvisual",
                mujoco_config.compiler_config.discardvisual,
                tag="compiler",
            )
        if mujoco_config.compiler_config.fusestatic is not None:
            world.update_attribute(
                "fusestatic", mujoco_config.compiler_config.fusestatic, tag="compiler"
            )
        if mujoco_config.compiler_config.eulerseq is not None:
            world.update_attribute(
                "eulerseq", mujoco_config.compiler_config.eulerseq, tag="compiler"
            )
        if mujoco_config.compiler_config.inertiagrouprange is not None:
            world.update_attribute(
                "inertiagrouprange",
                mujoco_config.compiler_config.inertiagrouprange,
                tag="compiler",
            )
        return world
    
    def import_scenario(self, scenario_config: ScenarioConfig):
        """Import a complete scenario into the simulation.
        
        Loads all elements defined in the scenario configuration including ground planes,
        robots, objects, and sensors into the current simulation environment.
        
        Args:
            scenario_config: Configuration object containing all scenario elements
            
        Raises:
            ValueError: If there is an error during scenario import
        """
        try:
            self.logger.log_info(f"Importing scenario: {scenario_config.name}")

            # Add ground planes
            for ground_plane_config in scenario_config.ground_planes:
                self.add_default_scene(ground_plane_config)

            # Add robots
            for robot_config in scenario_config.robots:
                self.add_robot(robot_config)

            # Add objects
            for object_config in scenario_config.objects:
                self.add_object(object_config)

            # Add sensors
            for sensor_config in scenario_config.sensors:
                self.add_sensor(sensor_config)

            self.logger.log_success(
                f"Scenario '{scenario_config.name}' imported successfully"
            )

        except Exception as e:
            self.logger.log_error(f"Error importing scenario: {str(e)}")
            raise ValueError(f"Error importing scenario: {str(e)}")

    def set_joint_positions(
        self,
        robot,
        joint_positions: list[float],
        joint_names: list[str],
        immediate: bool = False,
    ) -> None:
        """Set the joint positions for a robot.
        
        Args:
            robot: Robot instance or path string
            joint_positions: List of joint position values (in radians)
            joint_names: List of joint names to set positions for
            immediate: If True, update positions immediately without physics
            
        Raises:
            ValueError: If joint names or positions are invalid
            KeyError: If robot does not exist
        """
        return robot.set_joint_positions(
            positions=joint_positions, joint_names=joint_names, immediate=immediate
        )

    def get_joint_positions(self, robot, joint_names: list[str]) -> list[float]:
        """Get the current positions of specified joints on a robot.
        
        Args:
            robot: Robot instance or path string
            joint_names: List of joint names to get positions for
            
        Returns:
            list[float]: Current joint positions (in radians)
            
        Raises:
            KeyError: If robot does not exist
            ValueError: If joint names are invalid
        """
        return robot.get_joint_positions(joint_names=joint_names)

    def add_physics_callback(self, name: str, callback_fn: callable) -> None:
        """Register a callback function to be called during physics simulation.
        
        Args:
            name: Unique name for the callback
            callback_fn: Function to call (will receive simulator instance as argument)
            
        Raises:
            ValueError: If a callback with the same name already exists
        """
        if name in self._physics_callbacks:
            self.logger.log_warning(f"Callback with name '{name}' already exists")
            return
        self._physics_callbacks[name] = callback_fn

    def remove_physics_callback(self, name: str) -> None:
        """Remove a physics callback by its name.

        Args:
            name: Name of the callback to remove
        """
        if name not in self._physics_callbacks:
            self.logger.log_warning(f"Callback with name '{name}' does not exist")
            return
        del self._physics_callbacks[name]

    def physics_callback_exists(self, name: str) -> bool:
        """Check if a physics callback with the given name exists.

        Args:
            name: Name of the callback to check

        Returns:
            bool: True if callback exists, False otherwise
        """
        return name in self._physics_callbacks

    def get_robot_state(self, prim_path):
        """Get the current state of a robot in the simulation.
        
        Args:
            prim_path: Path to the robot
            
        Returns:
            dict: Robot state including position, orientation, joint positions, velocities, etc.
            
        Raises:
            KeyError: If robot does not exist at the specified path
        """
        robot_state = {}

        try:
            with self.lock:
                robot = self._robots.get(prim_path)["instance"]

                if robot is None:
                    self.logger.log_error(
                        f"Robot with key '{prim_path}' not found in robots dictionary."
                    )
                    self.logger.log_debug(
                        f"Available keys types: {[type(k) for k in self._robots.keys()]}"
                    )
                    raise KeyError(f"Robot with key '{prim_path}' not found")


                position = self.data.get_body_xpos(robot.root_body_name)
                quat_wxyz = self.data.get_body_xquat(robot.root_body_name)
                orientation = wxyz_to_xyzw(quat_wxyz)  # Convert to xyzw format

                # Get joint positions and velocities
                joint_positions = robot.get_all_joint_positions()
                joint_velocities = robot.get_all_joint_velocities()

                # Fill robot state
                robot_state["position"] = position
                robot_state["orientation"] = orientation
                robot_state["joint_positions"] = joint_positions
                robot_state["joint_velocities"] = joint_velocities

                return robot_state

        except Exception as e:
            self.logger.log_error(f"Error getting robot state: {e}")
            if "robot" in locals():
                self.logger.log_error(f"Robot type: {type(robot)}")
            raise

    def get_object_state(self, prim_path):
        """Get the current state of an object in the simulation.
        
        Args:
            prim_path: Path to the object
            
        Returns:
            dict: Object state including position, orientation, velocity, etc.
            
        Raises:
            KeyError: If object does not exist at the specified path
        """
        # Initialize object state
        object_state = {}

        # Try to get the object from _objects dictionary
        # This should work for both string keys and object instances
        try:
            with self.lock:
                obj = self._objects.get(prim_path)["instance"]

                if obj is None:
                    self.logger.log_error(
                        f"Object with key '{prim_path}' not found in objects dictionary."
                    )
                    self.logger.log_debug(
                        f"Available keys types: {[type(k) for k in self._objects.keys()]}"
                    )
                    raise KeyError(f"Object with key '{prim_path}' not found")

                # Get body name from object
                body_name = f"{obj.naming_prefix}main"
                
                # Get position and orientation directly from MuJoCo data
                position = self.data.get_body_xpos(body_name)
                quat_wxyz = self.data.get_body_xquat(body_name)
                orientation = wxyz_to_xyzw(quat_wxyz)  # Convert to xyzw format

                # Fill object state
                object_state["position"] = position
                object_state["orientation"] = orientation

                return object_state

        except Exception as e:
            self.logger.log_error(f"Error getting object state: {e}")
            if "obj" in locals():
                self.logger.log_error(f"Object type: {type(obj)}")
            raise

    @classmethod
    def get_synthnova_assets_directory(cls) -> str:
        """Retrieves the SynthNova assets directory path.

        This method retrieves the SynthNova assets directory path from the environment variable SYNTHNOVA_ASSETS.
        If the environment variable is not set, it raises a ValueError.
        If the environment variable is set, it returns the path.

        Returns:
            str: The path to the SynthNova assets directory.
        """
        # synthnova_assets_directory = os.getenv("SYNTHNOVA_ASSETS")
        synthnova_assets_directory = cls.get_root_directory() + "/assets"
        if synthnova_assets_directory is None:
            # raise ValueError(
            #     "SYNTHNOVA_ASSETS environment variable is not set. Have you run the asset pull script in README?"
            # )
            raise ValueError("assets directory not found")
        return synthnova_assets_directory

    @classmethod
    def get_root_directory(cls) -> str:
        """Retrieves the root directory path of the SynthNova Physics Simulator project.

        This method uses the current file's location to determine the project root directory.
        It traverses up three parent directories from the current file location to reach the
        project root.

        Returns:
            pathlib.Path: The absolute path to the project root directory.
        """
        return (pathlib.Path(__file__).parent.parent.parent.parent).as_posix()

    def export_scenario(
        self,
        file_path: str,
        scenario_name: str = None,
        scenario_description: str = None,
    ):
        """Export the current simulation state as a scenario configuration file.
        
        Creates a scenario configuration file from the current state of all entities
        in the simulation, including robots, objects, and sensors.
        
        Args:
            file_path: Path where the scenario file will be saved
            scenario_name: Name for the scenario (defaults to file name)
            scenario_description: Description of the scenario (optional)
            
        Returns:
            str: Path to the saved scenario file
            
        Raises:
            ValueError: If there is an error during export
        """
        try:
            # Create scenario config with proper initialization
            scenario_config = ScenarioConfig(
                name=scenario_name or pathlib.Path(file_path).stem,
                description=scenario_description
                or "Exported scenario from SynthNova Render Simulator",
                ground_planes=[
                    ground_plane["config"]
                    for ground_plane in self._ground_planes.values()
                ],
                robots=[robot["config"] for robot in self._robots.values()],
                objects=[obj["config"] for obj in self._objects.values() if isinstance(obj, dict) and "config" in obj],
                sensors=[sensor["config"] for sensor in self._sensors.values()],
            )
            # Export to file
            scenario_config.export_to_file(file_path)
            self.logger.log_info(f"Scenario exported to {file_path}")

        except Exception as e:
            self.logger.log_error(f"Error exporting scenario: {str(e)}")
            raise ValueError(f"Error exporting scenario: {str(e)}")

    def get_current_state(self) -> dict:
        """Get a complete snapshot of the current simulation state.
        
        Returns a dictionary containing the state of all robots and objects in the simulation,
        along with current physics parameters and simulation time.
        
        Returns:
            dict: Dictionary containing:
                - robots: State of each robot (positions, orientations, joint states)
                - objects: State of each object (positions, orientations)
                - physics_params: Current physics parameters
                - time: Current simulation time
        """
        with self.lock:
            state = {
                "robots": {},
                "objects": {},
                "physics_params": {},
                "time": self.data.time
            }
            
            # Export robot states
            for prim_path, robot_info in self._robots.items():
                robot_state = self.get_robot_state(prim_path)
                state["robots"][prim_path] = {
                    "position": robot_state["position"].tolist(),
                    "orientation": robot_state["orientation"].tolist(),
                    "joint_positions": robot_state["joint_positions"].tolist(),
                    "joint_velocities": robot_state["joint_velocities"].tolist()
                }
            
            # Export object states
            for prim_path, obj_info in self._objects.items():
                obj_state = self.get_object_state(prim_path)
                state["objects"][prim_path] = {
                    "position": obj_state["position"].tolist(),
                    "orientation": obj_state["orientation"].tolist()
                }
            
            # Export essential physics parameters
            state["physics_params"] = {
                "timestep": self.config.mujoco_config.timestep,
                "gravity": self.config.mujoco_config.gravity,
                "solver": self.config.mujoco_config.solver,
                "iterations": self.config.mujoco_config.iterations
            }
            
            return state
    
    def get_simulation_time(self) -> float:
        """Get current simulation time.
        
        Returns:
            float: Current simulation time
        """
        with self.lock:
            return self.data.time

    def get_step_count(self) -> int:
        """Get the number of simulation steps that have been executed.
        
        Returns:
            int: Number of simulation steps executed
        """
        with self.lock:
            return int(self.data.time / self.config.mujoco_config.timestep)

    def get_physics_dt(self) -> float:
        """Get the physics simulation timestep.
        
        Returns:
            float: The physics timestep in seconds
        """
        return self._mujoco_timestep if hasattr(self, '_mujoco_timestep') else 0.01
