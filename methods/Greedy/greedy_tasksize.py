import numpy as np
from env.config import VehicularEnvConfig
class Greedy(object):
    """ 贪心算法实现思路 """


    def __init__(self):
        self.config=VehicularEnvConfig()
        pass

    def choose_action(self, state,function) -> int:
        """ 根据任务队列选择合适的卸载节点 """
        action_list=[]
        State=state
        Function=function
        function_size=[]
        for i in range(self.config.rsu_number):
            function_size.append(Function[i].get_task_datasize())

        for i in range(len(function_size)):

            min_index = np.argmin(State)
            action_list.append(min_index+1)
            State[i]=State[i]+function_size[i]
        x=action_list[0]
        y=action_list[1]
        z=action_list[2]
        action= (x - 1) + (y - 1) * 14 + (z - 1) * 14 * 14 + 1
        return action
