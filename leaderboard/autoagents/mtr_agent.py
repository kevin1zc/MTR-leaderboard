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

from mtr.config import cfg, cfg_from_yaml_file
from mtr.models import model as model_utils
from mtr.utils import common_utils
from carla_api.utils.mtr_data_utils import create_scene_level_data, generate_prediction_dicts
from carla_api.agents.navigation.behavior_agent import BehaviorAgent

from carla_api.mpc.mpc_solver import MpcController
from carla_api.mpc.config import N, dt, MIN_TURN_SPEED, DEBUG_PRINTS, DEBUG_VISUALIZATION, DEBUG_TIMING

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
        self._dense_route = None  # Store the dense route (green dots, ~1m spacing)
        self._dense_route_parsed = False

        self._simulation_steps = 0
        self.last_control = None
        self.use_precomputed_waypoints = False
        self.follow_agent = False
        self.mpc_failure_count = 0  # Track consecutive MPC failures

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
        
        ego_bbox = self.player.bounding_box.extent
        self.ego_length = ego_bbox.x * 2.0
        self.ego_width = ego_bbox.y * 2.0
        print(f"[AGENT] Ego vehicle dimensions: length={self.ego_length:.2f}m, width={self.ego_width:.2f}m")

        self.temp_agent = BehaviorAgent(self.player, behavior='normal', opt_dict={
            'sampling_resolution': 0.24})

        self.mpc = MpcController(self.world, self.player, horizon=N, dt=dt)

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS

    def sensors(self):
        return [{'type': 'sensor.speedometer', 'id': 'Speed'}]

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        from leaderboard.utils.route_manipulation import downsample_route
        
        self._dense_route_world_coord = global_plan_world_coord
        ds_ids = downsample_route(global_plan_world_coord, 200)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        
        print(f"[AGENT] Received dense route: {len(global_plan_world_coord)} waypoints (~1m spacing)")
        print(f"[AGENT] Created sparse route: {len(self._global_plan_world_coord)} waypoints (~200m spacing)")
    
    def parse_route(self):
        self._route = []
        for pos, cmd in self._global_plan_world_coord:
            pos = [pos.location.x, pos.location.y]
            self._route.append(pos)
        self._route_parsed = True
        self._route = np.array(self._route)
    
    def parse_dense_route(self):
        self._dense_route = []
        for pos, cmd in self._dense_route_world_coord:
            pos = [pos.location.x, pos.location.y]
            self._dense_route.append(pos)
        self._dense_route_parsed = True
        self._dense_route = np.array(self._dense_route)
        print(f"[AGENT] Parsed dense route: {len(self._dense_route)} waypoints")

    def parse_carla_data(self, track_ids):
        info = {
            'scenario_id': "scenario_0",
            'timestamps_seconds': np.linspace(0.0, 9.0, 91),
            'current_time_index': 10,
            'sdc_track_index': 0,
            'objects_of_interest': [],
            'tracks_to_predict': {
                'track_index': list(range(len(track_ids))),
                'difficulty': [0] * len(track_ids),
                'object_type': ['TYPE_VEHICLE'] * len(track_ids)
            },
            'track_infos': self.decode_tracks(track_ids)
        }
        return info

    def decode_tracks(self, track_ids):
        track_infos = {'object_id': [], 'object_type': [], 'trajs': []}
        for i, object_id in enumerate(track_ids):
            trajs = self._trajectories[object_id]
            full_traj = np.zeros((11, 10))
            cur_traj = np.stack(trajs, axis=0)
            full_traj[-len(cur_traj):] = cur_traj
            track_infos['object_id'].append(object_id)
            track_infos['object_type'].append('TYPE_VEHICLE')
            track_infos['trajs'].append(full_traj)
        track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)
        return track_infos

    def get_ego_vehicle_state(self):
        transform = self.player.get_transform()
        x = transform.location.x
        y = transform.location.y
        yaw = np.deg2rad(transform.rotation.yaw)
        v = np.sqrt(self.player.get_velocity().x ** 2 +
                    self.player.get_velocity().y ** 2)

        return x, y, yaw, v
    
    def _visualize_trajectories(self, final_pred_dicts, prediction_horizon, base_z):
        if not DEBUG_VISUALIZATION:
            return
        
        # Ego vehicle color (bright cyan/light blue)
        ego_color = carla.Color(0, 255, 255)
        
        # Other vehicle colors
        colors = [
            carla.Color(255, 0, 0), carla.Color(255, 165, 0), carla.Color(255, 255, 0),
            carla.Color(0, 255, 0), carla.Color(0, 0, 255), carla.Color(75, 0, 130),
            carla.Color(238, 130, 238)
        ]
        
        # Visualize ego vehicle trajectory (index 0)
        if len(final_pred_dicts) > 0:
            ego_vehic_index = np.argmax(final_pred_dicts[0]['pred_scores'])
            ego_traj_full = final_pred_dicts[0]['pred_trajs'][ego_vehic_index][:prediction_horizon]
            
            for j in range(len(ego_traj_full)):
                traj_point = ego_traj_full[j]
                traj_location = carla.Location(x=float(traj_point[0]), y=float(traj_point[1]), z=base_z + 0.2)
                # Larger points for ego vehicle to make it more visible
                point_size = 0.15 if j < N else 0.10
                self.world.debug.draw_point(traj_location, size=point_size, color=ego_color, life_time=0.15)
                
                if j < len(ego_traj_full) - 1:
                    next_traj_point = ego_traj_full[j + 1]
                    next_location = carla.Location(x=float(next_traj_point[0]), y=float(next_traj_point[1]), z=base_z + 0.2)
                    # Thicker lines for ego vehicle
                    line_thickness = 0.15 if j < N else 0.08
                    self.world.debug.draw_line(traj_location, next_location, thickness=line_thickness,
                                              color=ego_color, life_time=0.15)
        
        # Visualize other vehicles' trajectories (starting from index 1)
        for i in range(1, len(final_pred_dicts)):
            temp_vehic_index = np.argmax(final_pred_dicts[i]['pred_scores'])
            temp_traj_full = final_pred_dicts[i]['pred_trajs'][temp_vehic_index][:prediction_horizon]
            color = colors[min(i-1, len(colors)-1)]
            
            for j in range(len(temp_traj_full)):
                traj_point = temp_traj_full[j]
                traj_location = carla.Location(x=float(traj_point[0]), y=float(traj_point[1]), z=base_z)
                point_size = 0.12 if j < N else 0.08
                self.world.debug.draw_point(traj_location, size=point_size, color=color, life_time=0.15)
                
                if j < len(temp_traj_full) - 1:
                    next_traj_point = temp_traj_full[j + 1]
                    next_location = carla.Location(x=float(next_traj_point[0]), y=float(next_traj_point[1]), z=base_z)
                    line_thickness = 0.1 if j < N else 0.05
                    self.world.debug.draw_line(traj_location, next_location, thickness=line_thickness,
                                              color=color, life_time=0.15)
    
    def _visualize_waypoints(self, closest_k_waypoint, x0, y0, yaw0, goal, base_z):
        if not DEBUG_VISUALIZATION or len(closest_k_waypoint) == 0:
            return
        
        target_wp = closest_k_waypoint[0]
        target_location = carla.Location(x=float(target_wp[0]), y=float(target_wp[1]), z=base_z + 1.0)
        self.world.debug.draw_point(target_location, size=0.3, color=carla.Color(255, 0, 255), life_time=0.15)
        self.world.debug.draw_line(
            carla.Location(x=x0, y=y0, z=base_z),
            carla.Location(x=float(target_wp[0]), y=float(target_wp[1]), z=base_z),
            thickness=0.2, color=carla.Color(255, 0, 255), life_time=0.15
        )
        
        goal_location = carla.Location(x=float(goal[0]), y=float(goal[1]), z=base_z + 1.5)
        self.world.debug.draw_point(goal_location, size=0.4, color=carla.Color(0, 255, 255), life_time=0.15)
    
    def _print_debug_info(self, x0, y0, yaw0, closest_k_waypoint, waypoints_, v0, reference_speed):
        if not DEBUG_PRINTS:
            return
        
        if len(closest_k_waypoint) > 0:
            target_wp = closest_k_waypoint[0]
            target_dist = np.linalg.norm(np.array([target_wp[0], target_wp[1]]) - np.array([x0, y0]))
            dx = target_wp[0] - x0
            dy = target_wp[1] - y0
            target_angle = np.arctan2(dy, dx)
            angle_diff = np.rad2deg(target_angle - yaw0)
            
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360
            
            print(f"[DEBUG] Ego pos: ({x0:.2f}, {y0:.2f}), yaw: {np.rad2deg(yaw0):.1f}°")
            print(f"[DEBUG] Target WP: ({target_wp[0]:.2f}, {target_wp[1]:.2f}), dist: {target_dist:.2f}m")
            print(f"[DEBUG] Target angle: {np.rad2deg(target_angle):.1f}°, Angle diff: {angle_diff:.1f}°")
        
        if waypoints_ and len(waypoints_) >= 3:
            print(f"[DEBUG MPC WPS] WP[0]: ({waypoints_[0][0]:.2f}, {waypoints_[0][1]:.2f}), "
                  f"WP[1]: ({waypoints_[1][0]:.2f}, {waypoints_[1][1]:.2f}), "
                  f"WP[2]: ({waypoints_[2][0]:.2f}, {waypoints_[2][1]:.2f})")
        
        print(f"[DEBUG] Current speed: {v0:.2f} m/s, Reference speed: {reference_speed:.2f} m/s")
    
    def _select_waypoints_for_mpc(self, current_location, heading, v0, x0, y0):
        dense_route_new, closest_index_dense = self.choose_ahead_waypoint(
            waypoints=self._dense_route, pos=current_location, heading=heading)
        
        if dense_route_new is False or len(dense_route_new) <= closest_index_dense:
            return []
        
        if v0 < 2.0:
            look_ahead_waypoints = 2
        elif v0 < 5.0:
            look_ahead_waypoints = 3
        else:
            look_ahead_waypoints = 4
        
        start_index = min(closest_index_dense + look_ahead_waypoints, len(dense_route_new) - 1)
        max_waypoints_for_mpc = min(50, len(dense_route_new) - start_index)
        return list(dense_route_new[start_index: start_index + max_waypoints_for_mpc])
    
    def _prepare_mpc_waypoints(self, closest_k_waypoint, x0, y0, yaw0=None):
        if len(closest_k_waypoint) == 0:
            print(f"[AGENT WARNING] No waypoints available, using current position")
            return [[x0, y0]] * N
        
        waypoints_ = []
        current_pos = np.array([x0, y0])
        current_heading = np.array([np.cos(yaw0), np.sin(yaw0)]) if yaw0 is not None else None
        
        for i, wp in enumerate(closest_k_waypoint[:N]):
            wp_pos = np.array([float(wp[0]), float(wp[1])])
            
            # Check if waypoint requires a sharp turn from current heading
            if current_heading is not None and i == 0:
                # Calculate direction to first waypoint from current position
                direction = wp_pos - current_pos
                if np.linalg.norm(direction) > 0.1:  # Avoid division by zero
                    direction_norm = direction / np.linalg.norm(direction)
                    # Calculate angle between current heading and waypoint direction
                    cos_angle = np.clip(np.dot(direction_norm, current_heading), -1.0, 1.0)
                    angle_change = np.arccos(cos_angle)
                    
                    # If waypoint requires sharp turn (> 30°), smooth it out
                    if np.rad2deg(angle_change) > 30.0:
                        if DEBUG_PRINTS:
                            print(f"[AGENT WARNING] First waypoint requires sharp turn ({np.rad2deg(angle_change):.1f}°), smoothing")
                        # Project waypoint direction onto current heading to smooth the turn
                        # Use a weighted combination: mostly current heading, some waypoint direction
                        weight_current = 0.7  # Prefer current heading
                        weight_waypoint = 0.3
                        smoothed_direction = (weight_current * current_heading + 
                                             weight_waypoint * direction_norm)
                        smoothed_direction = smoothed_direction / np.linalg.norm(smoothed_direction)
                        # Use distance to waypoint but in smoothed direction
                        distance = np.linalg.norm(direction)
                        wp_pos = current_pos + smoothed_direction * distance
            
            # Check for sharp turns between consecutive waypoints
            if i > 0 and len(waypoints_) > 0:
                prev_wp_pos = np.array(waypoints_[-1])
                direction = wp_pos - prev_wp_pos
                if np.linalg.norm(direction) > 0.1:
                    direction_norm = direction / np.linalg.norm(direction)
                    prev_direction = prev_wp_pos - (np.array(waypoints_[-2]) if len(waypoints_) > 1 else current_pos)
                    if np.linalg.norm(prev_direction) > 0.1:
                        prev_direction_norm = prev_direction / np.linalg.norm(prev_direction)
                        cos_angle = np.clip(np.dot(direction_norm, prev_direction_norm), -1.0, 1.0)
                        angle_change = np.arccos(cos_angle)
                        
                        # If waypoint requires sharp turn (> 30°), smooth it
                        if np.rad2deg(angle_change) > 30.0:
                            if DEBUG_PRINTS:
                                print(f"[AGENT WARNING] Waypoint {i} requires sharp turn ({np.rad2deg(angle_change):.1f}°), smoothing")
                            # Smooth the direction
                            weight_prev = 0.7
                            weight_new = 0.3
                            smoothed_direction = (weight_prev * prev_direction_norm + weight_new * direction_norm)
                            smoothed_direction = smoothed_direction / np.linalg.norm(smoothed_direction)
                            distance = np.linalg.norm(direction)
                            wp_pos = prev_wp_pos + smoothed_direction * distance
            
            waypoints_.append([float(wp_pos[0]), float(wp_pos[1])])
        
        # Fill remaining waypoints if needed
        if len(waypoints_) < N:
            last_wp = waypoints_[-1] if waypoints_ else [x0, y0]
            while len(waypoints_) < N:
                waypoints_.append(last_wp)
        
        return waypoints_
    
    def _collect_vehicle_data(self):
        ego_transform = self.player.get_transform()
        vehicles = self.world.get_actors().filter('vehicle.*')
        vehicle_dist = []
        
        def dist(loc):
            return math.sqrt((loc.x - ego_transform.location.x) ** 2 +
                           (loc.y - ego_transform.location.y) ** 2 +
                           (loc.z - ego_transform.location.z) ** 2)
        
        for vehicle in vehicles:
            transform = vehicle.get_transform()
            loc = transform.location
            yaw = transform.rotation.yaw
            vel = vehicle.get_velocity()
            
            if vehicle.id != self.player.id:
                vehicle_dist.append((dist(loc), vehicle.id, vehicle))
            
            dim = vehicle.bounding_box.extent * 2
            traj = np.array([loc.x, loc.y, loc.z, dim.x, dim.y, dim.z,
                           math.radians(yaw), vel.x, vel.y, 1])
            self._trajectories[vehicle.id].append(traj)
        
        vehicle_dist.sort()
        self._simulation_steps += 1
        
        track_ids = [self.player.id]
        for i in range(min(7, len(vehicle_dist))):
            track_ids.append(vehicle_dist[i][1])
        return track_ids
    
    def _run_mtr_prediction(self, track_ids):
        info = self.parse_carla_data(track_ids)
        info['vehicle_ids'] = track_ids
        info['map_infos'] = self.map_infos
        ret_infos = create_scene_level_data(info, cfg.DATA_CONFIG)
        
        batch_dict = {
            'batch_size': 1,
            'input_dict': ret_infos,
            'batch_sample_count': [len(info['vehicle_ids'])]
        }
        
        with torch.no_grad():
            batch_pred_dicts = self.model(batch_dict)
            final_pred_dicts = generate_prediction_dicts(batch_pred_dicts)[0]
        
        del batch_pred_dicts, ret_infos, batch_dict
        return final_pred_dicts
    
    def _build_dynamic_vehicle_list(self, final_pred_dicts, prediction_horizon):
        dyn_vehic_list = []
        for i in range(1, len(final_pred_dicts)):
            temp_vehic_index = np.argmax(final_pred_dicts[i]['pred_scores'])
            temp_traj_full = final_pred_dicts[i]['pred_trajs'][temp_vehic_index][:prediction_horizon]
            dyn_vehic_list.append(temp_traj_full[:N])
        return dyn_vehic_list

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

    def _check_oriented_bbox_collision(self, pos1, yaw1, length1, width1, pos2, yaw2, length2, width2, min_safety_distance):
        """
        Check if two oriented bounding boxes collide.
        Uses Separating Axis Theorem (SAT) for accurate collision detection.
        
        Args:
            pos1, pos2: Center positions of vehicles [x, y]
            yaw1, yaw2: Vehicle headings in radians
            length1, width1: Dimensions of vehicle 1
            length2, width2: Dimensions of vehicle 2
            min_safety_distance: Minimum safety distance between vehicles
        
        Returns:
            True if collision detected, False otherwise
        """
        # Calculate half-dimensions
        half_len1, half_wid1 = length1 / 2.0, width1 / 2.0
        half_len2, half_wid2 = length2 / 2.0, width2 / 2.0
        
        # Get rotation matrices
        cos1, sin1 = np.cos(yaw1), np.sin(yaw1)
        cos2, sin2 = np.cos(yaw2), np.sin(yaw2)
        
        # Get vehicle corners in local coordinates
        corners1_local = np.array([
            [-half_len1, -half_wid1],
            [half_len1, -half_wid1],
            [half_len1, half_wid1],
            [-half_len1, half_wid1]
        ])
        corners2_local = np.array([
            [-half_len2, -half_wid2],
            [half_len2, -half_wid2],
            [half_len2, half_wid2],
            [-half_len2, half_wid2]
        ])
        
        # Rotate and translate corners to world coordinates
        rot1 = np.array([[cos1, -sin1], [sin1, cos1]])
        rot2 = np.array([[cos2, -sin2], [sin2, cos2]])
        corners1 = corners1_local @ rot1.T + pos1
        corners2 = corners2_local @ rot2.T + pos2
        
        # Get perpendicular axes for SAT (normal vectors to edges)
        # For each box, we need axes perpendicular to two adjacent edges
        axes = []
        
        # Axes from box 1 (perpendicular to its edges)
        edge1_1 = corners1[1] - corners1[0]  # First edge
        edge1_2 = corners1[2] - corners1[1]  # Second edge
        if np.linalg.norm(edge1_1) > 1e-6:
            # Perpendicular axis (rotate 90 degrees)
            axis1 = np.array([-edge1_1[1], edge1_1[0]]) / np.linalg.norm(edge1_1)
            axes.append(axis1)
        if np.linalg.norm(edge1_2) > 1e-6:
            axis2 = np.array([-edge1_2[1], edge1_2[0]]) / np.linalg.norm(edge1_2)
            axes.append(axis2)
        
        # Axes from box 2 (perpendicular to its edges)
        edge2_1 = corners2[1] - corners2[0]
        edge2_2 = corners2[2] - corners2[1]
        if np.linalg.norm(edge2_1) > 1e-6:
            axis3 = np.array([-edge2_1[1], edge2_1[0]]) / np.linalg.norm(edge2_1)
            axes.append(axis3)
        if np.linalg.norm(edge2_2) > 1e-6:
            axis4 = np.array([-edge2_2[1], edge2_2[0]]) / np.linalg.norm(edge2_2)
            axes.append(axis4)
        
        # Project both boxes onto each axis and check for overlap
        for axis in axes:
            # Project corners of box 1
            proj1 = corners1 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            
            # Project corners of box 2
            proj2 = corners2 @ axis
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation (with safety distance)
            if max1 + min_safety_distance < min2 or max2 + min_safety_distance < min1:
                return False  # Separated on this axis, no collision
        
        return True  # No separation found, collision detected

    def _check_collision_avoidance(self, final_pred_dicts, track_ids, x0, y0, yaw0, current_speed):
        """
        Check for potential collisions in the next 1 second (10 prediction steps).
        Uses MTR predicted trajectories for both ego and other vehicles.
        MTR predictions are reliable within 1 second, beyond that they become less accurate.
        Uses oriented bounding box collision detection with minimum safety distance.
        Returns a safe speed (can be 0) if collision is detected.
        
        Args:
            final_pred_dicts: MTR prediction results for all vehicles (ego is first)
            track_ids: List of vehicle IDs (ego is first)
            x0, y0: Current ego vehicle position
            yaw0: Current ego vehicle heading (radians)
            current_speed: Current ego vehicle speed (m/s)
        
        Returns:
            safe_speed: Safe speed to avoid collision (0 if immediate collision risk)
        """
        from carla_api.mpc.config import dt
        
        # Don't apply collision avoidance if vehicle is already moving very slowly
        # This prevents getting stuck in a bad state where collision avoidance keeps slowing down
        # an already slow vehicle, causing erratic behavior
        if current_speed < 0.5:
            if DEBUG_PRINTS:
                print(f"[COLLISION DEBUG] Skipping collision check - vehicle speed too low ({current_speed:.2f} m/s)")
            return current_speed
        
        # Check next 1 second = 10 steps (dt = 0.1s)
        # MTR predictions are reliable within 1 second, beyond that accuracy degrades
        collision_check_horizon = 10
        
        # Minimum safety distance between vehicle bounding boxes (meters)
        # Reduced from 3.0m to 2.0m to reduce false positives
        # This still provides safe clearance while avoiding unnecessary slowdowns
        min_safety_distance = 2.0
        
        # Get ego vehicle dimensions
        ego_length = self.ego_length
        ego_width = self.ego_width
        
        # Get ego vehicle's MTR predicted trajectory (first 2 seconds)
        pred_ego = final_pred_dicts[0]
        ego_traj_index = np.argmax(pred_ego['pred_scores'])
        ego_pred_traj_full = pred_ego['pred_trajs'][ego_traj_index]  # Full predicted trajectory
        ego_pred_traj = ego_pred_traj_full[:collision_check_horizon]  # First 2 seconds
        
        # Estimate ego's heading from predicted trajectory points
        ego_pred_yaw = np.zeros(collision_check_horizon)
        for t in range(collision_check_horizon):
            if t == 0:
                # Use current heading for first step
                ego_pred_yaw[t] = yaw0
            elif t < len(ego_pred_traj):
                # Estimate heading from direction between consecutive points
                dx = ego_pred_traj[t, 0] - ego_pred_traj[t-1, 0]
                dy = ego_pred_traj[t, 1] - ego_pred_traj[t-1, 1]
                if np.sqrt(dx**2 + dy**2) > 0.01:  # Avoid division by zero
                    ego_pred_yaw[t] = np.arctan2(dy, dx)
                else:
                    ego_pred_yaw[t] = ego_pred_yaw[t-1]  # Keep previous heading
            else:
                ego_pred_yaw[t] = ego_pred_yaw[t-1]
        
        # Check collision with each nearby vehicle
        min_safe_speed = current_speed
        collision_detected = False
        
        for i in range(1, len(final_pred_dicts)):
            if i >= len(track_ids):
                continue
                
            vehicle_id = track_ids[i]
            
            # Get vehicle's best predicted trajectory
            vehicle_pred = final_pred_dicts[i]
            vehicle_traj_index = np.argmax(vehicle_pred['pred_scores'])
            vehicle_pred_traj = vehicle_pred['pred_trajs'][vehicle_traj_index][:collision_check_horizon]  # Shape: [20, 2]
            
            # Get vehicle dimensions and current heading from stored trajectory
            if vehicle_id in self._trajectories and len(self._trajectories[vehicle_id]) > 0:
                # Trajectory format: [x, y, z, dim.x, dim.y, dim.z, yaw, vel.x, vel.y, 1]
                latest_traj = self._trajectories[vehicle_id][-1]
                vehicle_length = latest_traj[3]  # dim.x
                vehicle_width = latest_traj[4]   # dim.y
                vehicle_current_yaw = latest_traj[6]  # yaw
            else:
                # Default vehicle dimensions if not available
                vehicle_length = 4.5
                vehicle_width = 2.0
                vehicle_current_yaw = 0.0
            
            # Estimate vehicle headings from trajectory points
            vehicle_pred_yaw = np.zeros(collision_check_horizon)
            for t in range(collision_check_horizon):
                if t == 0:
                    # Use current heading for first step
                    vehicle_pred_yaw[t] = vehicle_current_yaw
                elif t < len(vehicle_pred_traj):
                    # Estimate heading from direction between consecutive points
                    dx = vehicle_pred_traj[t, 0] - vehicle_pred_traj[t-1, 0]
                    dy = vehicle_pred_traj[t, 1] - vehicle_pred_traj[t-1, 1]
                    if np.sqrt(dx**2 + dy**2) > 0.01:  # Avoid division by zero
                        vehicle_pred_yaw[t] = np.arctan2(dy, dx)
                    else:
                        vehicle_pred_yaw[t] = vehicle_pred_yaw[t-1]  # Keep previous heading
                else:
                    vehicle_pred_yaw[t] = vehicle_pred_yaw[t-1]
            
            # Quick pre-filter: Check current distance to vehicle
            # Only check vehicles that are reasonably close (within 50m)
            current_vehicle_pos = vehicle_pred_traj[0] if len(vehicle_pred_traj) > 0 else None
            if current_vehicle_pos is not None:
                current_distance = np.linalg.norm(np.array([x0, y0]) - current_vehicle_pos)
                if current_distance > 50.0:  # Skip vehicles more than 50m away
                    continue
            
            # Check for collision at each time step
            for t in range(collision_check_horizon):
                if t >= len(ego_pred_traj) or t >= len(vehicle_pred_traj):
                    break
                
                ego_pos = ego_pred_traj[t]
                ego_yaw = ego_pred_yaw[t]
                vehicle_pos = vehicle_pred_traj[t]
                vehicle_yaw = vehicle_pred_yaw[t]
                
                # Quick distance check - skip if vehicles are far apart
                distance = np.linalg.norm(ego_pos - vehicle_pos)
                if distance > 30.0:  # Skip if more than 30m apart at this time step
                    continue
                
                # Check if vehicle is actually in front (longitudinal check)
                # Calculate relative position in ego's coordinate frame
                ego_heading_vec = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
                rel_pos = vehicle_pos - ego_pos
                longitudinal_dist = np.dot(rel_pos, ego_heading_vec)
                
                # Calculate lateral distance (perpendicular to ego heading)
                lateral_dist = abs(np.dot(rel_pos, np.array([-ego_heading_vec[1], ego_heading_vec[0]])))
                
                # Only care about vehicles that are in front or very close laterally
                # If vehicle is more than 20m behind, skip it
                if longitudinal_dist < -20.0:
                    continue
                
                # If vehicle is far laterally (more than 5m), it's likely in a different lane
                # Only check if it's also close longitudinally (within 15m)
                if lateral_dist > 5.0 and longitudinal_dist > 15.0:
                    continue
                
                # Check oriented bounding box collision only if vehicles are reasonably close
                if self._check_oriented_bbox_collision(
                    ego_pos, ego_yaw, ego_length, ego_width,
                    vehicle_pos, vehicle_yaw, vehicle_length, vehicle_width,
                    min_safety_distance
                ):
                    collision_detected = True
                    
                    # Calculate time to collision
                    # ego_pred_traj[t] represents position at time (t+1)*dt from now
                    time_to_collision = (t + 1) * dt
                    
                    # Debug: Print collision detection info
                    if DEBUG_PRINTS:
                        distance = np.linalg.norm(ego_pos - vehicle_pos)
                        print(f"[COLLISION DEBUG] Detected collision with vehicle {vehicle_id} at t={t}, "
                              f"time_to_collision={time_to_collision:.2f}s, distance={distance:.2f}m, "
                              f"ego_pos=({ego_pos[0]:.2f}, {ego_pos[1]:.2f}), "
                              f"vehicle_pos=({vehicle_pos[0]:.2f}, {vehicle_pos[1]:.2f})")
                    
                    # Since we only check 1 second ahead, all collisions detected are imminent
                    # Use aggressive speed reduction based on time to collision
                    # If collision is very soon (within 0.5 second), return minimum feasible speed
                    if time_to_collision < 0.5:
                        return 0.5  # Minimum speed to keep MPC feasible while allowing maximum braking
                    
                    # Otherwise, calculate a safe speed based on time to collision
                    # Use a more aggressive (quadratic) speed reduction curve
                    # The closer the collision, the slower we should go
                    # Formula: speed ratio uses quadratic curve from 0 (at 0.5s) to 1.0 (at 1.0s)
                    # This makes speed reduction more aggressive for imminent collisions
                    time_remaining = time_to_collision - 0.5  # Time remaining after 0.5s threshold
                    time_window = 0.5  # Window from 0.5s to 1.0s
                    
                    # Quadratic curve: ratio = (time_remaining / time_window)^2
                    # This makes early detection result in much slower speeds
                    safe_speed_ratio = max(0.0, min(1.0, (time_remaining / time_window) ** 2))
                    
                    # Additional safety: if collision is within 0.7 seconds, reduce speed more aggressively
                    if time_to_collision < 0.7:
                        safe_speed_ratio *= 0.5  # Further reduce speed by 50%
                    
                    safe_speed = current_speed * safe_speed_ratio
                    min_safe_speed = min(min_safe_speed, safe_speed)
        
        # If collision detected, return the minimum safe speed
        # Ensure minimum speed of 0.5 m/s to keep MPC problem feasible
        min_feasible_speed = 0.5
        if collision_detected:
            safe_speed = max(min_feasible_speed, min_safe_speed)
            if DEBUG_PRINTS:
                print(f"[COLLISION DEBUG] Collision detected, returning safe_speed={safe_speed:.2f} m/s "
                      f"(current_speed={current_speed:.2f} m/s)")
            return safe_speed
        
        # No collision detected, return current speed (no change needed)
        if DEBUG_PRINTS and len(final_pred_dicts) > 1:
            print(f"[COLLISION DEBUG] No collision detected, returning current_speed={current_speed:.2f} m/s")
        return current_speed

    def calculate_reference_speed(self, waypoints, current_speed):
        from carla_api.mpc.config import MAX_SPEED, MIN_TURN_SPEED
        
        if len(waypoints) < 3:
            return MAX_SPEED
        
        look_ahead = min(20, len(waypoints))
        max_curvature = 0.0
        
        for i in range(1, look_ahead - 1):
            p1 = np.array(waypoints[i-1])
            p2 = np.array(waypoints[i])
            p3 = np.array(waypoints[i+1])
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 0.1 or norm2 < 0.1:
                continue
            
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_change = np.arccos(cos_angle)
            curvature = angle_change / (norm1 + 0.01)
            max_curvature = max(max_curvature, curvature)
        
        if max_curvature < 0.05:
            target_speed = MAX_SPEED
        elif max_curvature > 0.3:
            target_speed = MIN_TURN_SPEED
        else:
            ratio = (max_curvature - 0.05) / (0.3 - 0.05)
            target_speed = MAX_SPEED - ratio * (MAX_SPEED - MIN_TURN_SPEED)
        
        max_speed_change_per_step = 2.0
        speed_error = target_speed - current_speed
        
        if abs(speed_error) <= max_speed_change_per_step:
            reference_speed = target_speed
        elif speed_error > 0:
            reference_speed = current_speed + max_speed_change_per_step
        else:
            reference_speed = current_speed - max_speed_change_per_step
        
        return max(MIN_TURN_SPEED, min(MAX_SPEED, reference_speed))

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        t_step_start = time.time()
        
        if self._simulation_steps % int(0.1 / self._delta_t) == 1:
            self._simulation_steps += 1
            return self.last_control
        
        start_location = self.player.get_location()
        x0, y0, yaw0, v0 = self.get_ego_vehicle_state()
        current_location = np.array([x0, y0])

        if not self._route_parsed:
            self.parse_route()
        if not self._dense_route_parsed:
            self.parse_dense_route()
        
        if self.use_precomputed_waypoints and timestamp < 0.1:
            self._route = self.precompute_parking_exit_path(self.world.get_map(), start_location=start_location)
            goal = self._route[1]
            self.temp_agent.set_destination(carla.Location(goal[0], goal[1], 0))
        else:
            goal = self._route[0] if len(self._route) == 1 else self._route[1]

            if self.follow_agent:
                self._route = None
                wp0 = self.world.get_map().get_waypoint(start_location)
                wpt = self.world.get_map().get_waypoint(carla.Location(goal[0], goal[1], 0))
                self.temp_agent.set_destination(carla.Location(goal[0], goal[1], 0))
                trace = self.temp_agent.trace_route(wp0, wpt)
                self._route = []
                for wp in trace:
                    self._route.append([wp[0].transform.location.x, wp[0].transform.location.y])
                self._route = np.array(self._route)
                self.follow_agent = False

        ego_transform = self.player.get_transform()
        fwd = ego_transform.get_forward_vector()
        heading = np.array([fwd.x, fwd.y])

        route_new, _ = self.choose_ahead_waypoint(waypoints=self._route, pos=current_location, heading=heading)
        if route_new is not False:
            self._route = route_new

        closest_k_waypoint = self._select_waypoints_for_mpc(current_location, heading, v0, x0, y0)
        track_ids = self._collect_vehicle_data()
        
        t_mtr_start = time.time()
        final_pred_dicts = self._run_mtr_prediction(track_ids)
        t_mtr = time.time() - t_mtr_start

        pred_ego = final_pred_dicts[0]
        traj_index = np.argmax(pred_ego['pred_scores'])
        ego_location = self.player.get_location()
        base_z = ego_location.z + 0.5
        
        prediction_horizon = min(30, final_pred_dicts[1]['pred_trajs'].shape[1]) if len(final_pred_dicts) > 1 else 30
        dyn_vehic_list = self._build_dynamic_vehicle_list(final_pred_dicts, prediction_horizon)
        self._visualize_trajectories(final_pred_dicts, prediction_horizon, base_z)

        waypoints_ = self._prepare_mpc_waypoints(closest_k_waypoint, x0, y0, yaw0)
        self._visualize_waypoints(closest_k_waypoint, x0, y0, yaw0, goal, base_z)
        
        # Check for collision avoidance
        # Use MTR predicted trajectories for both ego and other vehicles (first 2 seconds)
        collision_safe_speed = self._check_collision_avoidance(final_pred_dicts, track_ids, x0, y0, yaw0, v0)
        
        # Don't use ego's MTR predicted trajectory for static obstacle detection
        # MTR predictions can be wrong and cause MPC to make incorrect turns
        # Instead, use the waypoints themselves to find static obstacles
        # This ensures MPC checks obstacles along the path it's supposed to follow
        from carla_api.mpc.config import N
        # Use waypoints for static obstacle detection (they define the intended path)
        # Pad with current position if needed
        obstacle_check_path = np.array([[wp[0], wp[1]] for wp in waypoints_[:N]])
        if len(obstacle_check_path) < N:
            # Fill remaining with last waypoint
            last_wp = obstacle_check_path[-1] if len(obstacle_check_path) > 0 else np.array([x0, y0])
            padding = np.tile(last_wp, (N - len(obstacle_check_path), 1))
            obstacle_check_path = np.vstack([obstacle_check_path, padding])
        
        t_mpc_reset_start = time.time()
        self.mpc.reset_solver(x0, y0, yaw0, v0,
                              self.mpc.get_static_obstacles(obstacle_check_path),
                              self.mpc.get_static_obstacles_soft(obstacle_check_path),
                              waypoints_)
        t_mpc_reset = time.time() - t_mpc_reset_start

        # Calculate reference speed based on waypoints, then apply collision avoidance
        reference_speed = self.calculate_reference_speed(closest_k_waypoint, v0)
        # Apply collision avoidance: use the minimum of reference speed and collision-safe speed
        reference_speed = min(reference_speed, collision_safe_speed)
        
        # Ensure minimum speed to keep MPC problem feasible
        # MPC needs sufficient speed to solve the optimization problem
        # If speed is too low, MPC becomes infeasible, especially when turning
        from carla_api.mpc.config import MIN_SPEED
        min_feasible_speed = 1.0  # Minimum speed for MPC feasibility (increased from 0.3)
        reference_speed = max(reference_speed, min_feasible_speed)
        self._print_debug_info(x0, y0, yaw0, closest_k_waypoint, waypoints_, v0, reference_speed)
        
        t_mpc_update_start = time.time()
        self.mpc.update_cost_function(goal, dyn_vehic_list, reference_speed)
        t_mpc_update = time.time() - t_mpc_update_start
        
        t_mpc_solve_start = time.time()
        self.mpc.solve()
        t_mpc_solve = time.time() - t_mpc_solve_start

        if self.mpc.is_success:
            wheel_angle, acceleration = self.mpc.get_controls_value()
            throttle, brake, steer = self.mpc.process_control_inputs(wheel_angle, acceleration)
            control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
            self.mpc_failure_count = 0  # Reset failure count on success
            if DEBUG_PRINTS:
                print(f"[MPC SUCCESS] Steer: {steer:.3f} ({np.rad2deg(wheel_angle):.1f}°), Throttle: {throttle:.3f}, Brake: {brake:.3f}")
        else:
            self.mpc_failure_count += 1
            print(f"[AGENT] ✗✗✗ MPC FAILED ({self.mpc_failure_count} consecutive failures) - Falling back to BehaviorAgent")
            print(f"[AGENT] Ego state: x={x0:.2f}, y={y0:.2f}, yaw={np.rad2deg(yaw0):.1f}°, v={v0:.2f} m/s")
            print(f"[AGENT] Reference speed: {reference_speed:.2f} m/s")
            print(f"[AGENT] Waypoints: {len(waypoints_)} points")
            if len(waypoints_) > 0:
                print(f"[AGENT] First waypoint: ({waypoints_[0][0]:.2f}, {waypoints_[0][1]:.2f})")
            
            # If MPC fails repeatedly, increase reference speed to help recovery
            if self.mpc_failure_count > 5:
                print(f"[AGENT] Multiple MPC failures detected - attempting recovery with higher speed")
                # Try to recover by using BehaviorAgent with higher target speed
                self.temp_agent.set_destination(carla.Location(goal[0], goal[1], 0))
                control = self.temp_agent.run_step()
                # Increase throttle to help vehicle recover
                if control.throttle < 0.5:
                    control.throttle = min(0.8, control.throttle + 0.3)
            else:
                self.temp_agent.set_destination(carla.Location(goal[0], goal[1], 0))
                control = self.temp_agent.run_step()
            control.manual_gear_shift = False

        self.last_control = control
        
        if DEBUG_TIMING:
            t_step_total = time.time() - t_step_start
            print(f"\n[TIMING] run_step: {t_step_total*1000:.1f}ms | MTR: {t_mtr*1000:.1f}ms | MPC reset: {t_mpc_reset*1000:.1f}ms | MPC update: {t_mpc_update*1000:.1f}ms | MPC solve: {t_mpc_solve*1000:.1f}ms\n")

        return control
