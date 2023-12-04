import numpy as np
from typing import List # -> List[int] 指示应返回整数列表。


#获取任务属性的类：获取任务信息大小，计算能力：每bit所需转数，任务延迟约束
class Function(object):
    """任务属性及其操作"""
    #就是一个任务的三元数组

    def __init__(
            self,
            Function_task_datasize: float,
            Function_task_computing_resource: float,
            Function_task_delay: int

    ) -> None:
        self._Function_task_datasize = Function_task_datasize   #感知任务的大小
        self._Function_task_computing_resource = Function_task_computing_resource   #感知任务每bit的所需计算资源
        self._Function_task_delay = Function_task_delay #感知任务的延迟
    def get_task_datasize(self) -> float:
        return float(self._Function_task_datasize)

    def get_task_computing_resource(self) -> float:
        return float(self._Function_task_computing_resource)

    def get_task_delay(self) -> float:
        return float(self._Function_task_delay)



########################################################################
#操作队列的类：这个类的构造函数初始化了任务队列的属性，包括任务数量、任务大小范围、和随机数生成的种子
#功能有：获取任务列表，返回节点的总任务量，任务列表增加任务，自己产生任务（随着时间变化），处理任务（随着时间变化）
class TaskList(object):
    """节点上的待执行任务队列属性及操作"""

    def __init__(
            self,
            task_number: int,   #节点上待执行任务的个数
            minimum_datasize: float,   #节点上待执行任务大小的最小值
            maximum_datasize: float  #节点上待执行任务大小的最小值
            # seed: int
    ) -> None:
        self._task_number = task_number
        self._minimum_datasize = minimum_datasize
        self._maximum_datasize = maximum_datasize
        # self._seed=seed


        # 生成每个任务的数据量大小
        # np.random.seed(self._seed)  #设置种子确保每次使用相同的种子值运行代码时，都会得到相同的随机数序列。
        self._datasizes = np.random.uniform(self._minimum_datasize, self._maximum_datasize, self._task_number)
        #这一行生成一个数组（self._data_sizes），其中包含了在 _minimum_data_size 和 _maximum_data_size 之间均匀分布的随机浮点数。生成的随机值的数量等于 _task_number。
        self._task_list = [_ for _ in self._datasizes] #列表化

    def get_task_list(self) -> List[float]:
        return self._task_list

    def sum_datasize(self) -> float:
        """返回该节点的总任务量"""
        return sum(self._task_list)

    def add_task_list(self, new_data_size) -> None:
        """如果卸载到该节点，任务队列会增加"""
        self._task_list.append(new_data_size)

    def add_by_slot(self, task_number) -> None:
        """在时间转移中任务队列自动生成的任务"""
        data_sizes = np.random.uniform(self._minimum_datasize, self._maximum_datasize, task_number)
        for datasize in data_sizes:
            self._task_list.append(datasize)
            self._task_number += 1

    def delete_data_list(self, process_ability) -> None:
        """在时间转移中对任务队列中的任务进行处理"""
        while True:
            # 如果队列中没有任务
            if len(self._task_list) == 0:
                break
            # 如果队列中有任务
            elif process_ability >= self._task_list[0]:  # 单位时间计算能力大于数据量
                process_ability -= self._task_list[0]
                del self._task_list[0]
            else:  # 单位时间计算能力小于数据量
                self._task_list[0] -= process_ability
                break

########################################################################
#对每一辆车进行操作的类
#功能有：获取生存时间，获取位置坐标，存货时间-1，判断是否不再存活，获取车辆行驶速度，获取车辆计算能力，获取车辆任务队列，获取车辆任务队列里的任务量之和

class Vehicle(object):
    """车辆属性及其操作"""

    def __init__(
            self,
            road_range: int,    #马路长度
            min_vehicle_speed: float,  # 车辆最小行驶速度
            max_vehicle_speed: float,  # 车辆最大行驶速度
            min_task_number: float, #车辆队列中任务最小个数
            max_task_number: float, #车辆队列中任务最大个数
            min_task_datasize: float,  #每个任务大小的最大值
            max_task_datasize: float,  # 每个任务大小的最小值
            min_vehicle_compute_ability: float,  # 最小车辆计算能力
            max_vehicle_compute_ability: float,   #最大车辆计算能力
            vehicle_x_initial_location: list,  # 初始x坐标
            min_vehicle_y_initial_location: float,   #初始y坐标最小值
            max_vehicle_y_initial_location: float,# 初始y坐标最大值
            seed: int
    ) -> None:
        # 车辆在场景中的生存时间生成
        self._road_range = road_range
        self._seed=seed
        #生成初始y坐标
        self._min_vehicle_y_initial_location=min_vehicle_y_initial_location
        self._max_vehicle_y_initial_location = max_vehicle_y_initial_location
        np.random.seed(self._seed)
        self._vehicle_y_initial_location = np.random.randint(self._min_vehicle_y_initial_location, self._max_vehicle_y_initial_location, 1)[0]
        # y坐标
        self._vehicle_y_location=self._vehicle_y_initial_location
        # 生成初始x坐标
        np.random.seed(self._seed)
        self._vehicle_x_initial_location=np.random.choice(vehicle_x_initial_location)
        # x坐标
        self._vehicle_x_location = self._vehicle_x_initial_location
        #生成速度
        np.random.seed(self._seed)
        self._vehicle_speed = np.random.randint(min_vehicle_speed, max_vehicle_speed)  # 车辆速度
        if  self._vehicle_x_initial_location==0:
            self._vehicle_speed = self._vehicle_speed
        else:
            self._vehicle_speed = -self._vehicle_speed
        # 生成存活时间
        self._stay_time = int(self._road_range / self._vehicle_speed)

        # 车辆计算能力生成
        self._max_compute_ability = max_vehicle_compute_ability
        self._min_compute_ability = min_vehicle_compute_ability

        # np.random.seed(self._seed)
        self._min_compute_ability = np.random.uniform(self._min_compute_ability, self._max_compute_ability, 1)

        # 车辆任务队列生成
        self._min_task_number = min_task_number
        self._max_task_number = max_task_number
        self._max_datasize = max_task_datasize
        self._min_datasize = min_task_datasize
        # np.random.seed(self._seed)
        self._task_number = np.random.randint(self._min_task_number, self._max_task_number)
        self._vehicle_task_list = TaskList(self._task_number, self._min_datasize, self._max_datasize)
        # self._vehicle_task_list = TaskList(self._task_number, self._min_datasize, self._max_datasize, self._seed)



    #获取原始信息
    def get_initial_data(self) -> list:

        data = [self._vehicle_x_initial_location, self._vehicle_y_initial_location,self._vehicle_speed]
        return data

    # 生存时间相关
    #获取车辆生存时间
    def get_stay_time(self) -> int:
        return self._stay_time

    #获取当前坐标变化
    def get_location(self) -> list:
        location = [self._vehicle_x_location, self._vehicle_y_location]
        return location


    #每秒坐标变化
    def change_location(self) -> list:
        self._vehicle_x_location = self._vehicle_x_location + self._vehicle_speed*1
        self._vehicle_y_location =self._vehicle_y_initial_location
        location = [self._vehicle_x_location, self._vehicle_y_location]
        return location


    #车辆生存时间-1
    def decrease_stay_time(self) -> int:
        self._stay_time -= 1
        return self._stay_time

    def is_out(self) -> bool:
        if self._stay_time <= 5:  # 快要出去的车辆或者速度很大的车辆对任务不会感兴趣
            return True
        else:
            return False

    # 获取车辆行驶速度
    def get_vehicle_speed(self) -> float:
        return self._vehicle_speed

    #获取车辆计算速度
    def get_vehicle_compute_ability(self) -> float:
        return self._max_compute_ability

    # 车辆任务队列相关
    #获取车辆上的任务队列
    def get_task_list(self) -> TaskList:
        return self._vehicle_task_list

    # 获取车辆上的任务队列里所有任务的数据量之和
    def get_sum_tasks(self) -> float:
        if len(self._vehicle_task_list.get_task_list()) == 0:  # 车辆上没有任务
            return 0
        else:
            return self._vehicle_task_list.sum_datasize()  # 车辆上有任务



########################################################################
#对所有车辆进行操作的类
# 功能有：获取车辆数量，获取车辆基础信息列表，增加车辆数量，从车辆队列中删除不在范围内的车辆
class VehicleList(object):
    """实现场景中车辆的管理，包括车辆更新、停留时间更新以及任务队列更新"""

    def __init__(
            self,
            vehicle_number: int,    #车辆个数
            road_range: int,    #马路长度
            min_vehicle_speed: float,   #车辆最小行驶速度
            max_vehicle_speed: float,   #车辆最大行驶速度
            min_task_number: float,     #车辆队列中任务最小个数
            max_task_number: float,     #车辆队列中任务最大个数
            min_task_datasize: float,  #车辆队列中任务大小的最小值
            max_task_datasize: float,  #车辆队列中任务大小的最大值
            min_vehicle_compute_ability: float,   #车辆计算速度的最小值
            max_vehicle_compute_ability: float,   #车辆计算速度的最大值
            vehicle_x_initial_location: list,  # 初始x坐标
            min_vehicle_y_initial_location: float,  # 初始y坐标最小值
            max_vehicle_y_initial_location: float,  # 初始y坐标最大值
            seed: int


    ) -> None:
        self._seed = seed
        self._vehicle_number = vehicle_number
        self._road_range = road_range
        self._min_vehicle_speed = min_vehicle_speed
        self._max_vehicle_speed = max_vehicle_speed
        self._min_task_number = min_task_number
        self._max_task_number = max_task_number
        self._min_datasize = min_task_datasize
        self._max_datasize = max_task_datasize
        self._min_compute_ability = min_vehicle_compute_ability
        self._max_compute_ability = max_vehicle_compute_ability
        self._vehicle_x_initial_location=vehicle_x_initial_location
        self._min_vehicle_y_initial_location= min_vehicle_y_initial_location
        self._max_vehicle_y_initial_location= max_vehicle_y_initial_location


        #车辆基础信息列表，n辆车就有n个信息组
        self.vehicle_list = [
            Vehicle(
                road_range=self._road_range,
                min_vehicle_speed=self._min_vehicle_speed,
                max_vehicle_speed=self._max_vehicle_speed,
                min_task_number=self._min_task_number,
                max_task_number=self._max_task_number,
                min_task_datasize=self._min_datasize,
                max_task_datasize=self._max_datasize,
                min_vehicle_compute_ability=self._min_compute_ability,
                max_vehicle_compute_ability=self._max_compute_ability,
                vehicle_x_initial_location=self._vehicle_x_initial_location,  # 初始x坐标
                min_vehicle_y_initial_location=self._min_vehicle_y_initial_location,  # 初始y坐标最小值
                max_vehicle_y_initial_location=self._max_vehicle_y_initial_location,  # 初始y坐标最大值
                seed=self._seed+_
            )
            for _ in range(self._vehicle_number)]

    def get_vehicle_number(self) -> int:
        """获取车辆数量"""
        return self._vehicle_number

    def get_vehicle_list(self) -> List[Vehicle]:
        """获取车辆基础信息队列"""
        return self.vehicle_list

    def add_stay_vehicle(self, new_vehicle_number,time_now) -> None:
        """增加车辆数量"""
        # np.random.seed(self._seed)
        new_vehicle_list = [
            Vehicle(
                road_range=self._road_range,
                min_vehicle_speed=self._min_vehicle_speed,
                max_vehicle_speed=self._max_vehicle_speed,
                min_task_number=self._min_task_number,
                max_task_number=self._max_task_number,
                min_task_datasize=self._min_datasize,
                max_task_datasize=self._max_datasize,
                min_vehicle_compute_ability=self._min_compute_ability,
                max_vehicle_compute_ability=self._max_compute_ability,
                vehicle_x_initial_location=self._vehicle_x_initial_location,  # 初始x坐标
                min_vehicle_y_initial_location=self._min_vehicle_y_initial_location,  # 初始y坐标最小值
                max_vehicle_y_initial_location=self._max_vehicle_y_initial_location, # 初始y坐标最大值
                seed=time_now+_
            )
            for _ in range(new_vehicle_number)]

        self.vehicle_list = self.vehicle_list + new_vehicle_list
        self._vehicle_number += new_vehicle_number

    def delete_out_vehicle(self) -> None:
        """从队列中删除不在范围内的车辆"""
        i = 0
        while i < len(self.vehicle_list):
            if len(self.vehicle_list) == 0:#不一定需要判断
                pass
            elif self.vehicle_list[i].is_out():
                del self.vehicle_list[i]
                self._vehicle_number -= 1
            else:
                i += 1
########################################################################
#对某个RSU进行操作的类
#功能有：获取RSU计算能力，获取RSU任务队列，获取RSU任务队列上的任务量之和
class RSU(object):
    """RSU"""

    def __init__(
            self,
            min_task_number: float, #RSU队列中任务最小个数
            max_task_number: float, #RSU队列中任务最大个数
            min_task_datasize: float,  # RSU队列中任务大小的最小值
            max_task_datasize: float,  #RSU队列中任务大小的最大值
            min_rsu_compute_ability: float,  # RSU计算速度的最小值
            max_rsu_compute_ability: float #RSU计算速度的最大值
            # seed: int
    ) -> None:
        # rsu计算速度生成
        self._max_compute_ability = max_rsu_compute_ability
        self._min_compute_ability = min_rsu_compute_ability
        # self._seed = seed
        # np.random.seed(self._seed)
        self._compute_ability = np.random.uniform(self._min_compute_ability, self._max_compute_ability, 1)

        # 车辆任务队列生成
        self._min_task_number = min_task_number
        self._max_task_number = max_task_number
        self._max_datasize = max_task_datasize
        self._min_datasize = min_task_datasize
        # np.random.seed(self._seed)
        self._task_number = np.random.randint(self._min_task_number, self._max_task_number)
        self._rsu_task_list = TaskList(self._task_number, self._min_datasize, self._max_datasize)
        # self._rsu_task_list = TaskList(self._task_number, self._min_datasize, self._max_datasize, self._seed)

    #获取RSU计算速度
    def get_rsu_compute_ability(self) -> float:
        return self._compute_ability

    # 获取RSU任务队列
    def get_task_list(self) -> TaskList:
        return self._rsu_task_list

    # 获取RSU上的任务队列里所有任务的数据量之和
    def get_sum_tasks(self) -> float:
        if len(self._rsu_task_list.get_task_list()) == 0:  # RSU上没有任务
            return 0
        else:
            return self._rsu_task_list.sum_datasize()  # RSU上有任务



########################################################################
#对所有RSU进行操作的类
#获取RSU个数，获取RSU上的基础信息组
class RSUList(object):
    """RSU队列管理"""

    def __init__(
            self,
            rsu_number, #RSU个数
            min_task_number: float,  # RSU队列中任务最小个数
            max_task_number: float, #RSU队列中任务最大个数
            min_task_datasize: float,  # RSU队列中任务大小的最小值
            max_task_datasize: float,  #RSU队列中任务大小的最大值
            min_rsu_compute_ability: float,  # RSU计算速度的最小值
            max_rsu_compute_ability: float #RSU计算速度的最大值
            # seed: int
    ) -> None:
        # self._seed = seed
        self._rsu_number = rsu_number

        self._min_task_number = min_task_number
        self._max_task_number = max_task_number
        self._min_datasize = min_task_datasize
        self._max_datasize = max_task_datasize
        self._min_compute_ability = min_rsu_compute_ability
        self._max_compute_ability = max_rsu_compute_ability

        # 获取RSU类
        self.rsu_list = [
            RSU(
                min_task_number=self._min_task_number,
                max_task_number=self._max_task_number,
                min_task_datasize=self._min_datasize,
                max_task_datasize=self._max_datasize,
                min_rsu_compute_ability=self._min_compute_ability,
                max_rsu_compute_ability=self._max_compute_ability
                # seed=self._seed+_
            )
            for _ in range(rsu_number)]

    #获取RSU个数
    def get_rsu_number(self):
        return self._rsu_number
    #获取RSU的基础信息组
    def get_rsu_list(self):
        return self.rsu_list

#对时隙进行操作的类
class TimeSlot(object):
    """时隙属性及操作"""

    def __init__(self, start: int, end: int) -> None:
        self.start = start  #时间起始间隙
        self.end = end      #时间截止间隙
        self.slot_length = self.end - self.start    #时间间隙长度

        self.now = start    #当前时间间隙定位
        self.reset()    #做一些操作来将对象的属性或状态还原到初始状态

    def __str__(self):
        return f"now time: {self.now}, [{self.start} , {self.end}] with {self.slot_length} slots"

    #随着时间增加
    def add_time(self) -> None:
        """add time to the system"""
        self.now += 1

    #当前是否在时间截止间隙
    def is_end(self) -> bool:
        """check if the system is at the end of the time slots"""
        return self.now >= self.end
    #获取时间间隙长度
    def get_slot_length(self) -> int:
        """get the length of each time slot"""
        return self.slot_length
    #获取当前时间间隙定位
    def get_now(self) -> int:
        return self.now
    #重置
    def reset(self) -> None:
        self.now = self.start
