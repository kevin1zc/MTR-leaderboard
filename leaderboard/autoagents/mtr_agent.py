import math
import os
from collections import defaultdict, deque
from functools import partial
from pathlib import Path

import numpy as np
import pickle
import torch
import carla
import time

import psutil, gc, tracemalloc

from mtr.config import cfg, cfg_from_yaml_file
from mtr.models import model as model_utils
from mtr.utils import common_utils

# from carla_api.utils.world import World
# from carla_api.utils.carla_utils import HUD

from carla_api.utils.mtr_data_utils import create_scene_level_data, decode_map_features, generate_prediction_dicts
from carla_api.agents.navigation.behavior_agent import BehaviorAgent

from carla_api.mpc.mpc_solver import MpcController
from carla_api.mpc.config import N, dt

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from leaderboard.autoagents.autonomous_agent import Track


def get_entry_point():
    return 'MTRAgent'


class MTRAgent(AutonomousAgent):

    def __init__(self, carla_host, carla_port, debug=False):
        super().__init__(carla_host, carla_port, debug=False)
        self._trajectories = defaultdict(partial(deque, maxlen=11))
        self._delta_t = 0.05

        self._route_min_distance = 4.0
        self._route = None
        self._route_parsed = False

        self._simulation_steps = 0
        self.last_control = None
        self.use_precomputed_waypoints = True
        self.follow_agent = False

        mtr_dir = os.path.dirname(common_utils.__file__)

        cfg_path = os.path.join(mtr_dir, "../../tools/cfgs/waymo/mtr+100_percent_data.yaml")
        cfg_path = os.path.abspath(cfg_path)
        cfg_from_yaml_file(cfg_path, cfg)
        cfg.TAG = Path(cfg_path).stem
        cfg.EXP_GROUP_PATH = '/'.join(cfg_path.split('/')[1:-1])
        logger = common_utils.create_logger(None, rank=cfg.LOCAL_RANK)

        # log to file
        logger.info('**********************Start logging**********************')
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
        logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

        self.model = model_utils.MotionTransformer(config=cfg.MODEL)

        model_dir = os.path.join(mtr_dir, "../../carla_api/model/")
        model_dir = os.path.abspath(model_dir)

        map_name = CarlaDataProvider.get_world().get_map().name
        _, simple_name = os.path.split(map_name)

        map_info_path = os.path.join(model_dir, f'{simple_name}.pkl')

        with open(map_info_path, 'rb') as file:
            self.map_infos = pickle.load(file)

        model_path = os.path.join(model_dir, "latest_model.pth")
        model_path = os.path.abspath(model_path)

        self.model.load_params_from_file(model_path, logger=logger, to_cpu=False)
        self.model.cuda()
        self.model.eval()

        self.world = CarlaDataProvider.get_world()
        self.player = CarlaDataProvider.get_hero_actor()

        self.temp_agent = BehaviorAgent(self.player, behavior='normal', opt_dict={
            'sampling_resolution': 0.24})  # this may change later for trafic lights etc.

        self.mpc = MpcController(self.world, self.player, horizon=N, dt=dt)

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS  # At a minimum, this method sets the Leaderboard modality. In this case, SENSORS

    def sensors(self):
        # Add at least one sensor to get valid evaluation statistics
        sensors = [
            # {'type': 'sensor.camera.rgb', 'id': 'Center',
            #  'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': 300, 'height': 200,
            #  'fov': 100},
            # {'type': 'sensor.lidar.ray_cast', 'id': 'LIDAR',
            #  'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0},
            # {'type': 'sensor.other.radar', 'id': 'RADAR',
            #  'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0, 'fov': 30},
            # {'type': 'sensor.other.gnss', 'id': 'GPS',
            #  'x': 0.7, 'y': -0.4, 'z': 1.60},
            # {'type': 'sensor.other.imu', 'id': 'IMU',
            #  'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0},
            # {'type': 'sensor.opendrive_map', 'id': 'OpenDRIVE', 'reading_frequency': 1},
            {'type': 'sensor.speedometer', 'id': 'Speed'},
        ]
        return sensors

    def parse_route(self):
        self._route = []  # deque()
        for pos, cmd in self._global_plan_world_coord:
            pos = [pos.location.x, pos.location.y]  # np.array([pos.location.x, pos.location.y])
            self._route.append(pos)  # self._route.append((pos, cmd))

        self._route_parsed = True
        self._route = np.array(self._route)

    def parse_carla_data(self, track_ids):
        info = {}
        info['scenario_id'] = "scenario_0"
        info['timestamps_seconds'] = np.linspace(0.0, 9.0, 91)  # list of int of shape (91)
        info['current_time_index'] = 10  # int, 10
        info['sdc_track_index'] = 0  # set ego vehicle index to be 0
        info['objects_of_interest'] = []  # list, could be empty list

        info['tracks_to_predict'] = {
            'track_index': list(range(len(track_ids))),
            'difficulty': [0] * len(track_ids)
        }

        info['tracks_to_predict']['object_type'] = ['TYPE_VEHICLE'] * len(track_ids)

        info['track_infos'] = self.decode_tracks(track_ids)
        return info

    def decode_tracks(self, track_ids):
        track_infos = {
            'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            'object_type': [],
            'trajs': []
        }
        for i, object_id in enumerate(track_ids):
            trajs = self._trajectories[object_id]
            full_traj = np.zeros((11, 10))
            cur_traj = np.stack(trajs, axis=0)
            full_traj[-len(cur_traj):] = cur_traj
            track_infos['object_id'].append(object_id)
            track_infos['object_type'].append('TYPE_VEHICLE')
            track_infos['trajs'].append(full_traj)

        track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)  # (num_objects, num_timestamp, 9)

        return track_infos

    def get_ego_vehicle_state(self):
        transform = self.player.get_transform()
        x = transform.location.x
        y = transform.location.y
        yaw = np.deg2rad(transform.rotation.yaw)
        v = np.sqrt(self.player.get_velocity().x ** 2 +
                    self.player.get_velocity().y ** 2)

        return x, y, yaw, v

    def precompute_parking_exit_path(self, carla_map, start_location: carla.Location,
                                     spacing: float = 0.12):

        start_wp = carla_map.get_waypoint(start_location, lane_type=carla.LaneType.Driving)

        path = []
        current_wp = start_wp

        for i in range(150):
            next_wps = current_wp.next(spacing)
            if not next_wps:
                break

            candidate = next_wps[0]

            if candidate.lane_type == carla.LaneType.Driving:

                current_wp = candidate
                path.append((current_wp.transform.location.x,
                             current_wp.transform.location.y))
            else:
                continue

        return np.array(path)

    def choose_ahead_waypoint(self, waypoints, pos, heading):

        rel = waypoints - pos
        fronts = rel.dot(heading) > 0  # positive means ahead

        try:
            ahead_wps = waypoints[fronts]
            dists = np.linalg.norm(ahead_wps - pos, axis=1)
            return ahead_wps, np.argmin(dists)

        except:
            return False, False

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        """
            input_data: A dictionary containing sensor data for the requested sensors.
                        The data has been preprocessed at sensor_interface.py, and will be given as numpy arrays.
                        This dictionary is indexed by the ids defined in the sensor method.
            timestamp:  A timestamp of the current simulation instant (in seconds).
        """

        """
            Remember that you also have access to the route that the ego agent should travel to achieve its destination. 
            Use the self._global_plan member to access the geolocation route and self._global_plan_world_coord for 
            its world location counterpart.
        """

        """if timestamp < 2:
            self.last_control = carla.VehicleControl()
            return carla.VehicleControl()"""
        if self._simulation_steps % int(0.1 / self._delta_t) == 1:
            self._simulation_steps += 1
            return self.last_control
        start_location = self.player.get_location()

        x0, y0, yaw0, v0 = self.get_ego_vehicle_state()

        current_location = np.array([x0, y0])
        current_velocity = v0

        if current_velocity <= 5:
            multiplier = 1.2
        elif current_velocity <= 10:
            multiplier = 3.6
        elif current_velocity <= 20:
            multiplier = 6.2
        elif current_velocity <= 30:
            multiplier = 8
        elif current_velocity <= 40:
            multiplier = 10
        elif current_velocity <= 50:
            multiplier = 11
        elif current_velocity <= 60:
            multiplier = 12
        elif current_velocity <= 70:
            multiplier = 14
        else:
            multiplier = 14

        if self.use_precomputed_waypoints and timestamp < 0.1:
            self._route = self.precompute_parking_exit_path(self.world.get_map(), start_location=start_location)
            goal = self._route[1]
            self.temp_agent.set_destination(carla.Location(goal[0], goal[1], 0))
        else:
            if not self._route_parsed:
                self.parse_route()

            if len(self._route) == 1:
                goal = self._route[0]
            else:
                # goal_indx = min(20, self._route.shape[0])
                # goal = self._route[goal_indx-1]
                goal = self._route[1]
                # distance = np.linalg.norm(goal - cur_pos)
                # if distance < self._route_min_distance:
                #   self._route.popleft()

            if self.follow_agent:

                self._route = None

                wp0 = self.world.get_map().get_waypoint(start_location)
                wpt = self.world.get_map().get_waypoint(carla.Location(goal[0], goal[1], 0))

                self.temp_agent.set_destination(carla.Location(goal[0], goal[1], 0))
                trace = self.temp_agent.trace_route(wp0, wpt)
                self._route = []
                for wp in trace:
                    self._route.append([wp[0].transform.location.x,
                                        wp[0].transform.location.y])
                self._route = np.array(self._route)

                # multiplier = multiplier * 2

                self.follow_agent = False

        ego_transform = self.player.get_transform()
        fwd = ego_transform.get_forward_vector()
        heading = np.array([fwd.x, fwd.y])

        route_new, closest_index = self.choose_ahead_waypoint(waypoints=self._route, pos=current_location,
                                                              heading=heading)

        self._route = route_new

        if route_new is not False:
            termi = int(N * multiplier)
            closest_k_waypoint = list(route_new[closest_index + 1: closest_index + 1 + termi])
        else:
            closest_k_waypoint = list(route_new[-1])

        del route_new

        def dist(l):
            return math.sqrt((l.x - ego_transform.location.x) ** 2 + (l.y - ego_transform.location.y)
                             ** 2 + (l.z - ego_transform.location.z) ** 2)

        vehicles = self.world.get_actors().filter('vehicle.*')

        vehicle_dist = []

        for vehicle in vehicles:
            transform = vehicle.get_transform()
            loc = transform.location
            yaw = transform.rotation.yaw
            vel = vehicle.get_velocity()
            if vehicle.id != self.player.id:
                vehicle_dist.append((dist(loc), vehicle.id, vehicle))
            dim = vehicle.bounding_box.extent * 2
            traj = np.array([loc.x, loc.y, loc.z, dim.x, dim.y, dim.z, math.radians(yaw), vel.x, vel.y, 1])
            self._trajectories[vehicle.id].append(traj)
            self._simulation_steps = 0

        vehicle_dist.sort()

        track_ids = [self.player.id]
        for i in range(min(7, len(vehicle_dist))):  # find the 7 closest vehicles
            track_ids.append(vehicle_dist[i][1])

        self._simulation_steps += 1
        info = self.parse_carla_data(track_ids)
        info['vehicle_ids'] = track_ids

        info['map_infos'] = self.map_infos
        # info['dynamic_map_infos'] = self.dynamic_map_infos
        ret_infos = create_scene_level_data(info, cfg.DATA_CONFIG)

        batch_dict = {
            'batch_size': 1,
            'input_dict': ret_infos,
            'batch_sample_count': [len(info['vehicle_ids'])]
        }
        with torch.no_grad():
            batch_pred_dicts = self.model(batch_dict)
            final_pred_dicts = generate_prediction_dicts(batch_pred_dicts)[0]
            # print("final_pred_dicts: ", final_pred_dicts)
            # print("generate_prediction_dicts(batch_pred_dicts): ", generate_prediction_dicts(batch_pred_dicts))

        del batch_pred_dicts, ret_infos, batch_dict

        pred_ego = final_pred_dicts[0]
        traj_index = np.argmax(pred_ego['pred_scores'])
        # destination = pred_ego['pred_trajs'][traj_index][5]

        # Visualize predicted trajectory in CARLA (first 5 waypoints)
        # pred_trajs format: (num_modes, num_timestamps, 2) - only x, y coordinates
        pred_traj = pred_ego['pred_trajs'][traj_index]  # Shape: (num_timestamps, 2)
        num_waypoints_to_plot = min(10, len(pred_traj))
        
        if num_waypoints_to_plot > 0:
            # Convert to numpy array if needed for easier indexing
            pred_traj_array = np.array(pred_traj) if not isinstance(pred_traj, np.ndarray) else pred_traj
            
            # Get current vehicle location for z-coordinate reference
            ego_location = self.player.get_location()
            base_z = ego_location.z + 0.5  # Slightly above ground
            
            # Draw waypoints and connecting lines
            for i in range(num_waypoints_to_plot):
                waypoint = pred_traj_array[i + 10]
                # pred_trajs only has [x, y] coordinates (no z)
                if len(waypoint) >= 2:
                    x, y = float(waypoint[0]), float(waypoint[1])
                    location = carla.Location(x=x, y=y, z=base_z)
                    
                    # Draw point at waypoint location (gold color)
                    self.world.debug.draw_point(
                        location, 
                        size=0.1,  # Increased size for better visibility
                        color=carla.Color(255, 215, 0),  # Gold
                        life_time=0.2  # Lasts 1 second for better visibility
                    )
                    
                    # Draw line connecting to next waypoint
                    if i < num_waypoints_to_plot - 1:
                        next_waypoint = pred_traj_array[i + 1]
                        if len(next_waypoint) >= 2:
                            next_x, next_y = float(next_waypoint[0]), float(next_waypoint[1])
                            next_location = carla.Location(x=next_x, y=next_y, z=base_z)
                            
                            # Draw line connecting waypoints (gold color)
                            self.world.debug.draw_line(
                                location,
                                next_location,
                                thickness=0.1,  # Increased thickness for better visibility
                                color=carla.Color(255, 215, 0),  # Gold
                                life_time=0.2  # Lasts 1 second for better visibility
                            )

        dyn_vehic_list = []
        for i in range(1, len(final_pred_dicts)):
            temp_vehic_index = np.argmax(final_pred_dicts[i]['pred_scores'])
            temp_traj = final_pred_dicts[i]['pred_trajs'][temp_vehic_index][: 10]
            dyn_vehic_list.append(temp_traj)

        x0, y0, yaw0, v0 = self.get_ego_vehicle_state()

        temp_list = []

        if isinstance(closest_k_waypoint[-1], np.float64):
            temp_list.append(closest_k_waypoint)
            waypoints_ = temp_list * 10
        else:
            waypoints_ = closest_k_waypoint if len(closest_k_waypoint) >= 10 else closest_k_waypoint + \
                                                                                  [closest_k_waypoint[-1]] * (10 - len(
                closest_k_waypoint))

        self.mpc.reset_solver(x0, y0, yaw0, v0,
                              self.mpc.get_static_obstacles(np.array(pred_ego['pred_trajs'][traj_index][: N])), \
                              self.mpc.get_static_obstacles_soft(np.array(pred_ego['pred_trajs'][traj_index][: N])),
                              waypoints_)

        waypoints_ = None

        self.mpc.update_cost_function(goal, dyn_vehic_list)
        self.mpc.solve()

        if self.mpc.is_success:
            print("is_success : True")
            wheel_angle, acceleration = self.mpc.get_controls_value()
            throttle, brake, steer = self.mpc.process_control_inputs(wheel_angle, acceleration)
            control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

        else:
            print("is_success : False")
            self.temp_agent.set_destination(carla.Location(goal[0], goal[1], 0))
            control = self.temp_agent.run_step()
            control.manual_gear_shift = False

        self.last_control = control

        """
        input_data: A dictionary containing sensor data for the requested sensors.
                    The data has been preprocessed at sensor_interface.py, and will be given as numpy arrays.
                    This dictionary is indexed by the ids defined in the sensor method.
        timestamp:  A timestamp of the current simulation instant.
        """

        """
        Remember that you also have access to the route that the ego agent should travel to achieve its destination. 
        Use the self._global_plan member to access the geolocation route and self._global_plan_world_coord for 
        its world location counterpart.
        """

        return control
