from typing import Optional, Union, List, Tuple

import gym
from gym import spaces
from gym.core import RenderFrame, ObsType
from env.datastruct import VehicleList, RSUList, TimeSlot, Function
import torch
from env.config import VehicularEnvConfig
import math
import numpy as np
from scipy import integrate  # 求积分用到的
import copy
import random
import time

import networkx as nx


class RoadState(gym.Env):
    """updating"""

    def __init__(
            self,
            env_config: Optional[VehicularEnvConfig] = None,
            time_slot: Optional[TimeSlot] = None,
            vehicle_list: Optional[VehicleList] = None,
            rsu_list: Optional[RSUList] = None
    ):
        self.config = env_config or VehicularEnvConfig()  # 环境参数
        self.timeslot = time_slot or TimeSlot(start=self.config.time_slot_start, end=self.config.time_slot_end)
        self.rsu_number = self.config.rsu_number
        self.vehicle_number = self.config.vehicle_number
        self.rsu_range = self.config.rsu_range
        self.vehicle_range = self.config.vehicle_range
        self.seed = self.config.seed

        # 车辆与RSU的初始化，此处只是初始化了这两个类
        self.vehicle_list = vehicle_list or VehicleList(
            vehicle_number=self.vehicle_number,
            road_range=self.config.road_range,
            min_vehicle_speed=self.config.min_vehicle_speed,
            max_vehicle_speed=self.config.max_vehicle_speed,
            min_task_number=self.config.min_vehicle_task_number,  # 车辆队列中任务最小个数
            max_task_number=self.config.max_vehicle_task_number,  # 车辆队列中任务最大个数
            min_task_datasize=self.config.min_task_datasize,  # 车辆队列中任务大小的最小值
            max_task_datasize=self.config.max_task_datasize,  # 车辆队列中任务大小的最大值
            min_vehicle_compute_ability=self.config.min_vehicle_compute_ability,  # 车辆计算速度的最小值
            max_vehicle_compute_ability=self.config.max_vehicle_compute_ability,  # 车辆计算速度的最大值
            vehicle_x_initial_location=self.config.vehicle_x_initial_location,  # 初始x坐标
            min_vehicle_y_initial_location=self.config.min_vehicle_y_initial_location,  # 初始y坐标最小值
            max_vehicle_y_initial_location=self.config.max_vehicle_y_initial_location,  # 初始y坐标最大值
            seed=self.seed
        )

        self.rsu_list = rsu_list or RSUList(
            rsu_number=self.rsu_number,
            min_task_number=self.config.min_rsu_task_number,  # RSU队列中任务最小个数
            max_task_number=self.config.max_rsu_task_number,  # RSU队列中任务最大个数
            min_task_datasize=self.config.min_task_datasize,  # RSU队列中任务大小的最小值
            max_task_datasize=self.config.max_task_datasize,  # RSU队列中任务大小的最大值
            min_rsu_compute_ability=self.config.min_rsu_compute_ability,  # RSU计算速度的最小值
            max_rsu_compute_ability=self.config.max_rsu_compute_ability  # RSU计算速度的最大值
            # seed=self.seed
        )

        self.action_space = spaces.Discrete(self.config.action_size)
        self.observation_space = spaces.Box(low=self.config.low, high=self.config.high, dtype=np.float32)
        # 具体来说，spaces.Discrete(self._config.action_size) 创建了一个离散空间对象，其中 self._config.action_size 指定了该离散空间中可能的状态数量。
        # 因此，这行代码的作用是初始化 action_space 变量，并将其设置为表示可能动作的离散空间。在这个空间中，每个动作都是一个整数，范围从0到 self._config.action_size-1。
        self.state = None
        self.reward = 0
        self.function = None

    def _state_perception(self) -> np.ndarray:
        """ 这只是一个读取操作，在执行动作之前的队列情况"""
        vehicle_state = [vehicle.get_sum_tasks() for vehicle in self.vehicle_list.vehicle_list]
        rsu_state = [rsu.get_sum_tasks() for rsu in self.rsu_list.rsu_list]

        self.state = np.concatenate([vehicle_state, rsu_state])
        # np.concatenate()用于将两个或多个数组沿指定的轴（维度）连接在一起，创建一个新的数组。
        return np.array(self.state, dtype=np.float32)

    def _function_generator(self) -> List[Function]:
        """ 产生我们关注的任务 """
        new_function = []

        for i in range(self.rsu_number):
            # np.random.seed(self.seed + i)

            Function_task_datasize = np.random.uniform(self.config.Function_min_task_datasize,
                                                       self.config.Function_max_task_datasize)
            Function_task_delay = int(
                np.random.uniform(self.config.Function_min_task_delay, self.config.Function_max_task_delay))

            function = Function(Function_task_datasize, self.config.Function_task_computing_resource,
                                Function_task_delay)
            new_function.append(function)

        return new_function

    def _reset_road(self) -> None:
        """ 重置RSU队列，车辆队列 """
        self.vehicle_list = VehicleList(
            vehicle_number=self.vehicle_number,
            road_range=self.config.road_range,
            min_vehicle_speed=self.config.min_vehicle_speed,
            max_vehicle_speed=self.config.max_vehicle_speed,
            min_task_number=self.config.min_vehicle_task_number,  # 车辆队列中任务最小个数
            max_task_number=self.config.max_vehicle_task_number,  # 车辆队列中任务最大个数
            min_task_datasize=self.config.min_task_datasize,  # 车辆队列中任务大小的最小值
            max_task_datasize=self.config.max_task_datasize,  # 车辆队列中任务大小的最大值
            min_vehicle_compute_ability=self.config.min_vehicle_compute_ability,  # 车辆计算速度的最小值
            max_vehicle_compute_ability=self.config.max_vehicle_compute_ability,  # 车辆计算速度的最大值
            vehicle_x_initial_location=self.config.vehicle_x_initial_location,  # 初始x坐标
            min_vehicle_y_initial_location=self.config.min_vehicle_y_initial_location,  # 初始y坐标最小值
            max_vehicle_y_initial_location=self.config.max_vehicle_y_initial_location,  # 初始y坐标最大值
            seed=self.seed
        )
        self.rsu_list = RSUList(
            rsu_number=self.rsu_number,
            min_task_number=self.config.min_rsu_task_number,  # RSU队列中任务最小个数
            max_task_number=self.config.max_rsu_task_number,  # RSU队列中任务最大个数
            min_task_datasize=self.config.min_task_datasize,  # RSU队列中任务大小的最小值
            max_task_datasize=self.config.max_task_datasize,  # RSU队列中任务大小的最大值
            min_rsu_compute_ability=self.config.min_rsu_compute_ability,  # RSU计算速度的最小值
            max_rsu_compute_ability=self.config.max_rsu_compute_ability  # RSU计算速度的最大值
            # seed=self.seed
        )

        for i, vehicle in enumerate(self.vehicle_list.vehicle_list):
            # np.random.seed(self.seed + i)
            num_location_changes = np.random.randint(0, 16)

            for _ in range(num_location_changes):
                vehicle.change_location()

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            # 这是一个可选参数，它允许你指定一个随机数种子。种子用于控制伪随机数生成器的行为，如果你想要在每次重置时获得相同的随机状态，可以设置种子。通常用于复现实验结果。如果不提供种子，则默认为 None，表示不使用特定的种子。
            return_info: bool = False,
            # 这是一个布尔值参数，它确定是否要在 reset 方法中返回额外的信息。如果设置为 True，则 reset 方法可能会返回一些关于环境状态的额外信息或元数据，以便在需要时进行分析或记录。如果设置为 False，则只返回环境状态。默认为 False。
            options: Optional[dict] = None,
            # 这是一个可选的字典参数，用于传递其他配置选项。字典可以包含任何其他与 reset 方法相关的配置信息，具体取决于你的应用程序或环境的需求。如果不需要任何额外的配置选项，可以将其设置为 None。
    ):
        self.timeslot.reset()  # 重置时间
        self._reset_road()  # 重置道路
        self.function = self._function_generator()  # 新任务
        self.state = self._state_perception()  # 读取状态
        return np.array(self.state, dtype=np.float32),self.function

    # 获取所有RSU坐标
    def all_rsu_location(self):
        rsu_location = []
        group_size = self.config.road_range // self.config.rsu_number
        road_width_plus_5 = self.config.road_width + 5  # 为了避免多次计算

        for i in range(self.config.rsu_number):
            group_start = i * group_size
            middle = group_start + group_size // 2
            rsu_location.append((middle, road_width_plus_5))

        return rsu_location

    # 获取所有车辆速度
    def all_vehicle_speed(self):
        return [vehicle.get_vehicle_speed() for vehicle in self.vehicle_list.vehicle_list]

    # 获取所有车辆坐标
    def all_vehicle_location(self):
        return [vehicle.get_location() for vehicle in self.vehicle_list.vehicle_list]

    # 获取所有车辆坐标随时间的变化
    def change_all_vehicle_location(self):
        for vehicle in self.vehicle_list.vehicle_list:
            vehicle.change_location()

    # 获取r2v连通时间
    def r2v_connect_time(self, rsu_location, vehicle_location, vehicle_speed):
        rsu_distance_squared = self.rsu_range ** 2 - (rsu_location[1] - vehicle_location[1]) ** 2

        if vehicle_speed > 0:
            connect_time = (math.sqrt(rsu_distance_squared) + (rsu_location[0] - vehicle_location[0])) / abs(
                vehicle_speed)
        else:
            connect_time = (math.sqrt(rsu_distance_squared) - (rsu_location[0] - vehicle_location[0])) / abs(
                vehicle_speed)

        return connect_time

    def v2v_connect_time(self, vehicle1_location, vehicle2_location, vehicle1_speed, vehicle2_speed):
        v1_speed = abs(vehicle1_speed)
        v2_speed = abs(vehicle2_speed)
        rsu_range_squared = self.vehicle_range ** 2
        distance_squared = (vehicle1_location[1] - vehicle2_location[1]) ** 2

        if (vehicle1_speed > 0 and vehicle2_speed > 0) or (vehicle1_speed < 0 and vehicle2_speed < 0):
            if vehicle1_speed > 0 and vehicle2_speed > 0:
                if v1_speed > v2_speed:
                    connect_time = (math.sqrt(rsu_range_squared - distance_squared) + (
                            vehicle2_location[0] - vehicle1_location[0])) / abs(v1_speed - v2_speed)
                elif v1_speed == v2_speed:
                    connect_time = 10000  # 速度相等，一直联通，取值为10000
                else:
                    connect_time = (math.sqrt(rsu_range_squared - distance_squared) - (
                            vehicle2_location[0] - vehicle1_location[0])) / abs(v1_speed - v2_speed)
            else:
                if v1_speed > v2_speed:
                    connect_time = (math.sqrt(rsu_range_squared - distance_squared) - (
                            vehicle2_location[0] - vehicle1_location[0])) / abs(v1_speed - v2_speed)
                elif v1_speed == v2_speed:
                    connect_time = 10000
                else:
                    connect_time = (math.sqrt(rsu_range_squared - distance_squared) + (
                            vehicle2_location[0] - vehicle1_location[0])) / abs(v1_speed - v2_speed)
        else:
            if (vehicle1_speed > 0 and vehicle2_speed < 0):
                connect_time = (math.sqrt(rsu_range_squared - distance_squared) + (
                        vehicle2_location[0] - vehicle1_location[0])) / (v1_speed + v2_speed)
            else:
                connect_time = (math.sqrt(rsu_range_squared - distance_squared) - (
                        vehicle2_location[0] - vehicle1_location[0])) / (v1_speed + v2_speed)

        return connect_time

    # 获取r2v的传输速率

    def r2v_rate(self, rsu_location, vehicle_location, vehicle_speed):
        def rate(t):
            distance = math.sqrt((rsu_location[0] - (vehicle_location[0] + vehicle_speed * t)) ** 2 + (
                    rsu_location[1] - vehicle_location[1]) ** 2)
            return self.config.r2v_B * math.log2(1 + (self.config.rsu_p * self.config.k) / (self.config.w * distance))

        connect_time = self.r2v_connect_time(rsu_location, vehicle_location, vehicle_speed)
        time_points = np.linspace(0, connect_time, num=1000)  # 创建用于积分的时间点
        rates = [rate(t) for t in time_points]  # 计算速率
        average_rate = integrate.simps(rates, time_points) / connect_time  # 使用辛普森积分法计算平均速率
        return average_rate

    # 获取r2v的传输速率

    def v2v_rate(self, vehicle1_location, vehicle2_location, vehicle1_speed, vehicle2_speed):
        def rate(t):
            delta_x = (vehicle1_location[0] + vehicle1_speed * t) - (vehicle2_location[0] + vehicle2_speed * t)
            distance = math.sqrt(delta_x ** 2 + (vehicle1_location[1] - vehicle2_location[1]) ** 2)

            if distance == 0:
                rate_value = self.config.v2v_B * math.log2(
                    1 + (self.config.vehicle_p * self.config.k) / (self.config.w * 1))
            else:
                rate_value = self.config.v2v_B * math.log2(
                    1 + (self.config.vehicle_p * self.config.k) / (self.config.w * distance))
            return rate_value

        connect_time = self.v2v_connect_time(vehicle1_location, vehicle2_location, vehicle1_speed, vehicle2_speed)
        time_points = np.linspace(0, connect_time, num=1000)  # 创建用于积分的时间点
        rates = [rate(t) for t in time_points]  # 计算速率
        average_rate = integrate.simps(rates, time_points) / connect_time  # 使用辛普森积分法计算平均速率
        return average_rate

    # 获取最优路径以及该路径的连通时间
    def find_optional_paths(self, rsu_location, all_vehicle_location, all_vehicle_speed, rsu_range, vehicle_range,
                            Vehicle_id):
        # 创建一个无向图表示网络拓扑
        G = nx.Graph()

        # 添加RSU节点
        G.add_node(0, pos=rsu_location, speed=0)
        # 添加车辆节点
        for vehicle_id, (vehicle_location, vehicle_speed) in enumerate(zip(all_vehicle_location, all_vehicle_speed),
                                                                       start=1):
            G.add_node(vehicle_id, pos=vehicle_location, speed=vehicle_speed)

        # 添加边，如果两个节点之间的距离在通信范围内
        for node1 in G.nodes():
            for node2 in G.nodes():
                if node1 == node2:
                    continue
                pos1 = G.nodes[node1]["pos"]
                pos1_speed = G.nodes[node1]["speed"]
                pos2 = G.nodes[node2]["pos"]
                pos2_speed = G.nodes[node2]["speed"]

                distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
                if distance <= rsu_range or (node1 != 0 and node2 != 0 and distance <= vehicle_range):
                    if node1 != 0 and node2 != 0 and distance <= vehicle_range:
                        communication_time = self.v2v_connect_time(pos1, pos2, pos1_speed, pos2_speed)
                        G.add_edge(node1, node2, weight=communication_time)
                    else:
                        if node1 == 0:
                            communication_time = self.r2v_connect_time(pos1, pos2, pos2_speed)
                            G.add_edge(node1, node2, weight=communication_time)
                        else:
                            communication_time = self.r2v_connect_time(pos2, pos1, pos1_speed)
                            G.add_edge(node1, node2, weight=communication_time)

        # 检查指定的车是否与RSU通信，如果不通信则返回False
        if Vehicle_id not in G.neighbors(0):
            return False

        def dfs_paths(graph, start, end, path=[]):
            path = path + [start]
            if start == end:
                return [path]
            paths = []
            for node in graph.neighbors(start):
                if node not in path:
                    new_paths = dfs_paths(graph, node, end, path)
                    for new_path in new_paths:
                        paths.append(new_path)
            return paths

        rsu_id = 0
        vehicle_id = Vehicle_id  # 你要查询的车辆编号
        paths = dfs_paths(G, vehicle_id, rsu_id)

        # 输出路径及路径中边的权值（通信时间）
        result_paths = []
        result_weights = []
        for path in paths:
            weighted_path = [(node, G[path[i]][path[i + 1]]['weight']) for i, node in enumerate(path[:-1])]
            nodes, weights = zip(*weighted_path)
            result_paths.append(list(nodes) + [path[-1]])
            min_weight = min(weights)  # 找到路径中的最小权值
            result_weights.append(min_weight)

        # 找到最大权值的路径
        max_weight_index = result_weights.index(max(result_weights))
        optimal_path = result_paths[max_weight_index]
        optimal_weight = result_weights[max_weight_index]

        return optimal_path, optimal_weight

    # # 定义映射函数，将 (x, y, z) 映射为整数
    # def composite_action(x, y, z):
    #     return (x - 1) + (y - 1) * 14 + (z - 1) * 14 * 14+1

    # 分解动作,3个rsu,3个动作
    def decomposition_action(self, action):
        action=action+1#一般算法的小标从0开始，我们的环境从1开始
        action_number=self.rsu_number+self.vehicle_number+1
        z = ((action - 1) // (action_number * action_number)) + 1
        y = (((action - 1) // action_number) % action_number) + 1
        x = ((action - 1) % action_number) + 1
        action_list = [x, y, z]
        return action_list


    def _function_allocation(self, action,function_generator) -> None:
        """放置任务"""
        action_list = self.decomposition_action(action)
        new_functions = function_generator # 生成新任务，只调用一次

        for i, entity_index in enumerate(action_list):
            if 1 <= entity_index <= self.vehicle_number:
                entity = self.vehicle_list.get_vehicle_list()[entity_index - 1]
            elif self.vehicle_number + 1 <= entity_index <= (self.vehicle_number + self.rsu_number):
                entity = self.rsu_list.get_rsu_list()[entity_index - self.vehicle_number - 1]
            else:
                continue

            task_datasize = new_functions[i].get_task_datasize()
            entity.get_task_list().add_task_list(task_datasize)

    def _update_road(self, action,function_generator) -> object:
        """更新道路状态"""
        self.timeslot.add_time()  # 当前时隙 now+1
        now = self.timeslot.get_now()

        # 放置任务
        self._function_allocation(action,function_generator)

        # 更新车辆位置
        self.change_all_vehicle_location()

        # 更新任务队列和车辆
        for vehicle in self.vehicle_list.get_vehicle_list():
            vehicle.decrease_stay_time()  # 生存时间-1
            process_ability = copy.deepcopy(vehicle.get_vehicle_compute_ability())  # 获取计算能力
            process_ability = (process_ability * (10 ** 6)) / (
                        self.config.Function_task_computing_resource * 8 * 1024 * 1024)
            vehicle.get_task_list().delete_data_list(process_ability)  # 每个时隙都会处理队列里的任务量
            vehicle.get_task_list().add_by_slot(1)  # 每个时隙车辆都会自动生成一个任务

        for rsu in self.rsu_list.get_rsu_list():
            process_ability = copy.deepcopy(rsu.get_rsu_compute_ability())  # 获取计算能力
            process_ability=( process_ability*(10**6))/(self.config.Function_task_computing_resource*8*1024*1024)
            rsu.get_task_list().delete_data_list(process_ability)  # 每个时隙都会处理队列里的任务量
            rsu.get_task_list().add_by_slot(2)  # 每个时隙车辆都会自动生成一个任务

        # 判断是否要删除车辆
        self.vehicle_list.delete_out_vehicle()
        vehicle_number_now = self.config.vehicle_number - self.vehicle_list.get_vehicle_number()

        # 更新车辆
        if vehicle_number_now > 0:
            # self.vehicle_list.add_stay_vehicle(vehicle_number_now,self.timeslot.get_now())
            self.vehicle_list.add_stay_vehicle(vehicle_number_now, random.randint(1, 100))

        return self.vehicle_list, self.rsu_list

    # 获取奖励函数
    def get_reward(self, action,function_generator):
        action_list = self.decomposition_action(action)
        T_every_task_list = []#存储每个奖励消耗的时间
        offloading_vehicle = 0
        offloading_rsu = 0
        offloading_cloud = 0
        not_complete_number = 0
        for i in range(len(action_list)):  # i：0，1，2
            t_U = 0
            t_W = 0
            t_P = 0

            rate_flag = 1  # 判断传输速度限制的标记
            # 卸载到车辆
            if action_list[i] <= self.vehicle_number:
                result = self.find_optional_paths(self.all_rsu_location()[i], self.all_vehicle_location(),
                                                  self.all_vehicle_speed(), self.rsu_range,
                                                  self.vehicle_range, action_list[i])
                # print(result)#测试用*****************************************
                if result == False:
                    T_every_task_list.append(self.config.punishment)
                    not_complete_number+=1
                    # print(i, "to-Vehicle", self.config.punishment)
                else:
                    Path = result[0]
                    Connect_time = result[1]
                    # 求传输时延
                    if len(Path) == 2:
                        transfer_rate = self.r2v_rate(self.all_rsu_location()[i],
                                                      self.all_vehicle_location()[action_list[i] - 1],
                                                      self.all_vehicle_speed()[action_list[i] - 1])

                        if transfer_rate < self.config.min_transfer_rate:
                            rate_flag = 0
                        t_U = function_generator[i].get_task_datasize() / transfer_rate
                    else:
                        transfer_rate_list = []

                        for j in range(len(Path) - 1, 0, -1):
                            if j == len(Path) - 1:
                                transfer_rate = self.r2v_rate(self.all_rsu_location()[i],
                                                              self.all_vehicle_location()[Path[j - 1] - 1],
                                                              self.all_vehicle_speed()[Path[j - 1] - 1])
                                if transfer_rate < self.config.min_transfer_rate:
                                    rate_flag = 0
                                transfer_rate_list.append(transfer_rate)
                            else:
                                transfer_rate = self.v2v_rate(self.all_vehicle_location()[Path[j] - 1],
                                                              self.all_vehicle_location()[Path[j] - 1],
                                                              self.all_vehicle_speed()[Path[j] - 1],
                                                              self.all_vehicle_speed()[Path[j - 1] - 1])
                                if transfer_rate < self.config.min_transfer_rate:
                                    rate_flag = 0
                                transfer_rate_list.append(transfer_rate)

                        for k in range(len(transfer_rate_list)):
                            t_U = t_U +function_generator[i].get_task_datasize() / transfer_rate_list[k]

                    # 求等待时延
                    t_W = (self.vehicle_list.get_vehicle_list()[action_list[
                                                                    i] - 1].get_sum_tasks() * 8 * 1024 * 1024 * self.config.Function_task_computing_resource) \
                          / (self.vehicle_list.get_vehicle_list()[action_list[i] - 1].get_vehicle_compute_ability() * (
                                10 ** 6))

                    for m in range(len(action_list)):
                        if (action_list[m] == action_list[i]) and (m < i):
                            t_W = t_W + (function_generator[
                                             m].get_task_datasize() * 8 * 1024 * 1024 * self.config.Function_task_computing_resource) \
                                  / (self.vehicle_list.get_vehicle_list()[
                                         action_list[i] - 1].get_vehicle_compute_ability() * (10 ** 6))

                    # 求计算时延
                    t_P = (function_generator[
                               i].get_task_datasize() * 8 * 1024 * 1024 * self.config.Function_task_computing_resource) \
                          / (self.vehicle_list.get_vehicle_list()[action_list[i] - 1].get_vehicle_compute_ability() * (
                                10 ** 6))
                    t_all = -(t_U + t_W + t_P)

                    if (-t_all > Connect_time) or (-t_all > function_generator[i].get_task_delay()) or (
                            rate_flag == 0):
                        not_complete_number += 1
                        t_all = self.config.punishment
                        offloading_vehicle -= 1

                    # print(i,"to-Vehicle", t_all)
                    T_every_task_list.append(t_all)
                    offloading_vehicle+=1
            # 卸载到RSU
            elif action_list[i] <= (self.vehicle_number + self.rsu_number):
                # 求传输时延
                if (action_list[i] - self.vehicle_number) == (i + 1):
                    t_U = 0
                else:
                    t_U = abs(action_list[i] - self.vehicle_number - i - 1) * self.config.r2r_onehop_time
                # print("to-RSU", t_U)
                # 求等待时延
                t_W = (self.rsu_list.get_rsu_list()[action_list[
                                                        i] - self.vehicle_number - 1].get_sum_tasks() * 8 * 1024 * 1024 * self.config.Function_task_computing_resource) \
                      / (self.rsu_list.get_rsu_list()[
                             action_list[i] - self.vehicle_number - 1].get_rsu_compute_ability() * (10 ** 6))
                for m in range(len(action_list)):
                    if (action_list[m] == action_list[i]) and (m < i):
                        t_W = t_W + (function_generator[
                                         m].get_task_datasize() * 8 * 1024 * 1024 * self.config.Function_task_computing_resource) \
                              / (self.rsu_list.get_rsu_list()[
                                     action_list[i] - self.vehicle_number - 1].get_rsu_compute_ability() * (10 ** 6))
                # print("to-RSU",t_W)
                # 求计算时延
                t_P = (function_generator[
                           i].get_task_datasize() * 8 * 1024 * 1024 * self.config.Function_task_computing_resource) \
                      / (self.rsu_list.get_rsu_list()[
                             action_list[i] - self.vehicle_number - 1].get_rsu_compute_ability() * (10 ** 6))
                t_all = -(t_U + t_W + t_P)
                # print("to-RSU", t_P)
                if (-t_all > self.config.rsu_connect_time) or (-t_all > function_generator[i].get_task_delay()):
                    not_complete_number += 1
                    t_all = self.config.punishment
                    offloading_rsu -= 1
                T_every_task_list.append(t_all)
                offloading_rsu+=1
                # print(i,"to-RSU", t_all)
            # 卸载到cloud
            else:
                # 求传输时延
                t_U = function_generator[i].get_task_datasize() / self.config.c2r_rate
                # 求等待时延
                t_W = 0
                # 求计算时延
                t_P=0
                # t_P = (function_generator[
                #            i].get_task_datasize() * 8 * 1024 * 1024 * self.config.Function_task_computing_resource) \
                #       / (self.config.cloud_compute_ability * (10 ** 6))
                t_all = -(t_U + t_W + t_P)

                if (-t_all > self.config.cloud_connect_time) or (-t_all > function_generator[i].get_task_delay()):
                    not_complete_number += 1
                    t_all = self.config.punishment
                    offloading_cloud -= 1
                T_every_task_list.append(t_all)
                offloading_cloud+=1
                # print(i,"to-cloud",t_all)
        # print( T_every_task_list)
        t_all_task = float(sum(T_every_task_list))
        complete_number = len(action_list)-not_complete_number
        return t_all_task,offloading_vehicle,offloading_rsu,offloading_cloud,complete_number

    def step(self, action,function_generator):

        # 奖励值更新
        reward,offloading_vehicle,offloading_rsu,offloading_cloud,complete_number = self.get_reward(action,function_generator)

        # done更新
        done = self.timeslot.is_end()

        # 更新产生的任务
        function = self._function_generator()

        # 每个时隙更新的任务数据
        self.vehicle_list, self.rsu_list = self._update_road(action,function_generator)

        # 所有车和RSU的情况，包括生存时间，任务队列等
        obs_vehicle = [float(vehicle.get_sum_tasks()) for vehicle in self.vehicle_list.get_vehicle_list()]
        obs_rsu = [float(rsu.get_sum_tasks()) for rsu in self.rsu_list.get_rsu_list()]
        self.state = np.array(obs_vehicle + obs_rsu, dtype=np.float32)
        # print(self.all_vehicle_location())
        return self.state, reward, done,function,offloading_vehicle,offloading_rsu,offloading_cloud,complete_number

    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def close(self):
        pass
