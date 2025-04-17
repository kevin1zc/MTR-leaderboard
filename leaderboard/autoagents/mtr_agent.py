import math
import os
from collections import defaultdict, deque
from functools import partial
from pathlib import Path

import numpy as np
import pickle
import torch
import carla

from mtr.config import cfg, cfg_from_yaml_file
from mtr.models import model as model_utils
from mtr.utils import common_utils
from carla_api.utils.mtr_data_utils import create_scene_level_data, decode_map_features, generate_prediction_dicts
from carla_api.agents.navigation.behavior_agent import BehaviorAgent

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from leaderboard.autoagents.autonomous_agent import Track


def get_entry_point():
    return 'MTRAgent'


class MTRAgent(AutonomousAgent):

    def __init__(self, carla_host, carla_port, debug=False):
        super().__init__(carla_host, carla_port, debug=False)
        self._delta_t = 0.05
        self._trajectories = defaultdict(partial(deque, maxlen=11))
        self._simulation_steps = 0
        self._last_control = None

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

        map_info_path = os.path.join(model_dir, 'map_infos.pkl')
        dynamic_map_info_path = os.path.join(model_dir, 'dynamic_map_infos.pkl')

        with open(map_info_path, 'rb') as file:
            self.map_infos = pickle.load(file)
        with open(dynamic_map_info_path, 'rb') as file:
            self.dynamic_map_infos = pickle.load(file)

        model_path = os.path.join(model_dir, "latest_model.pth")
        model_path = os.path.abspath(model_path)

        self.model.load_params_from_file(model_path, logger=logger, to_cpu=False)
        self.model.cuda()
        self.model.eval()

        self.agent = None

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

        if timestamp < 2:
            return carla.VehicleControl()

        if self._simulation_steps % int(0.1 / self._delta_t) == 1:
            return self._last_control

        world = CarlaDataProvider.get_world()
        player = CarlaDataProvider.get_hero_actor()

        # Trying to set up a Carla autopilot. Can ignore it in MPC controller.
        if self.agent is None:
            self.agent = BehaviorAgent(player, behavior="normal", opt_dict={'sampling_resolution': 1.0})

        vehicles = world.get_actors().filter('vehicle.*')

        # actors = CarlaDataProvider.get_actors()
        ego_transform = player.get_transform()

        def dist(l):
            return math.sqrt((l.x - ego_transform.location.x) ** 2 + (l.y - ego_transform.location.y)
                             ** 2 + (l.z - ego_transform.location.z) ** 2)

        vehicle_dist = []

        for vehicle in vehicles:
            transform = vehicle.get_transform()
            loc = transform.location
            yaw = transform.rotation.yaw
            vel = vehicle.get_velocity()
            if vehicle.id != player.id:
                vehicle_dist.append((dist(loc), vehicle.id, vehicle))
            dim = vehicle.bounding_box.extent * 2
            traj = np.array([loc.x, loc.y, loc.z, dim.x, dim.y, dim.z, math.radians(yaw), vel.x, vel.y, 1])
            self._trajectories[vehicle.id].append(traj)
            self._simulation_steps = 0

        vehicle_dist.sort()

        track_ids = [player.id]
        for i in range(min(7, len(vehicle_dist))):  # find the 7 closest vehicles
            track_ids.append(vehicle_dist[i][1])

        self._simulation_steps += 1
        info = self.parse_carla_data(track_ids)
        info['vehicle_ids'] = track_ids

        info['map_infos'] = self.map_infos
        info['dynamic_map_infos'] = self.dynamic_map_infos
        ret_infos = create_scene_level_data(info, cfg.DATA_CONFIG)
        batch_dict = {
            'batch_size': 1,
            'input_dict': ret_infos,
            'batch_sample_count': [len(info['vehicle_ids'])]
        }
        with torch.no_grad():
            batch_pred_dicts = self.model(batch_dict)
            final_pred_dicts = generate_prediction_dicts(batch_pred_dicts)[0]

        pred_ego = final_pred_dicts[0]
        traj_index = np.argmax(pred_ego['pred_scores'])
        destination = pred_ego['pred_trajs'][traj_index][5]
        self.agent.set_destination(
            carla.Location(destination[0].item(), destination[1].item(), ego_transform.location.z))

        try:
            control = self.agent.run_step()
        except:
            destination = pred_ego['pred_trajs'][traj_index][30]
            self.agent.set_destination(
                carla.Location(destination[0].item(), destination[1].item(), ego_transform.location.z))
            control = self.agent.run_step()
        control.manual_gear_shift = False

        self._last_control = control

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
