import sys, os
import numpy as np
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
print(curr_path)
import matplotlib.pyplot as plt
import seaborn as sns
from env.utils import plot_A3C_rewards,plot_contrast_rewards
from env.utils import plot_A3C_completion_rate,plot_contrast_completion_rate
from env.utils import plot_A3C_completion_rate_lr,plot_A3C_rewards_lr
from env.utils import plot_different_tasksize_average_rewards
from env.utils import plot_different_vehicle_number_average_rewards
from env.utils import plot_different_vehicle_speed_average_rewards
from env.utils import plot_different_task_computation_resource_average_rewards
from env.utils import plot_different_vehicle_speed_average_rewards_1
from env.utils import plot_different_vehicle_number_average_rewards_1
#
# #绘制A3C实验奖励收敛图
A3C_train_rewards_2_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/2processes/20231009-092121/results/train_rewards.npy")
A3C_train_rewards_2_processes.tolist()
A3C_train_ma_rewards_2_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/2processes/20231009-092121/results/train_ma_rewards.npy")
A3C_train_ma_rewards_2_processes.tolist()

A3C_train_rewards_4_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/4processes/20231007-083233/results/train_rewards.npy")
A3C_train_rewards_4_processes.tolist()
A3C_train_ma_rewards_4_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/4processes/20231007-083233/results/train_ma_rewards.npy")
A3C_train_ma_rewards_4_processes.tolist()


A3C_train_rewards_5_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/5processes/20231007-083546/results/train_rewards.npy")
A3C_train_rewards_5_processes.tolist()
A3C_train_ma_rewards_5_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/5processes/20231007-083546/results/train_ma_rewards.npy")
A3C_train_ma_rewards_5_processes.tolist()

A3C_train_rewards_6_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/6processes/20231005-153224/results/train_rewards.npy")
A3C_train_rewards_6_processes.tolist()
A3C_train_ma_rewards_6_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/6processes/20231005-153224/results/train_ma_rewards.npy")
A3C_train_ma_rewards_6_processes.tolist()

plot_A3C_rewards(A3C_train_ma_rewards_2_processes,A3C_train_ma_rewards_4_processes,A3C_train_ma_rewards_6_processes)

#绘制A3C实验完成率收敛图
A3C_train_completion_rate_2_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/2processes/20231009-092121/results/train_completion_rate.npy")
A3C_train_completion_rate_2_processes.tolist()
A3C_train_ma_completion_rate_2_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/2processes/20231009-092121/results/train_ma_completion_rate.npy")
A3C_train_ma_completion_rate_2_processes.tolist()


A3C_train_completion_rate_4_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/4processes/20231007-083233/results/train_completion_rate.npy")
A3C_train_completion_rate_4_processes.tolist()
A3C_train_ma_completion_rate_4_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/4processes/20231007-083233/results/train_ma_completion_rate.npy")
A3C_train_ma_completion_rate_4_processes.tolist()

A3C_train_completion_rate_5_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/5processes/20231007-083546/results/train_completion_rate.npy")
A3C_train_completion_rate_5_processes.tolist()
A3C_train_ma_completion_rate_5_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/5processes/20231007-083546/results/train_ma_completion_rate.npy")
A3C_train_ma_completion_rate_5_processes.tolist()

A3C_train_completion_rate_6_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/6processes/20231005-153224/results/train_completion_rate.npy")
A3C_train_completion_rate_6_processes.tolist()
A3C_train_ma_completion_rate_6_processes = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/6processes/20231005-153224/results/train_ma_completion_rate.npy")
A3C_train_ma_completion_rate_6_processes.tolist()

plot_A3C_completion_rate(A3C_train_ma_completion_rate_2_processes,A3C_train_ma_completion_rate_4_processes,A3C_train_ma_completion_rate_6_processes)


# #绘制不同学习率A3C实验奖励收敛图

A3C_train_rewards_1 = np.load("C:/Users/23928/Desktop/结果/0.002/20231124-103556/results/train_rewards.npy")
A3C_train_rewards_1.tolist()
A3C_train_ma_rewards_1 = np.load("C:/Users/23928/Desktop/结果/0.002/20231124-103556/results/train_ma_rewards.npy")
A3C_train_ma_rewards_1.tolist()

A3C_train_rewards_2 = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/6processes/20231005-153224/results/train_rewards.npy")
A3C_train_rewards_2.tolist()
A3C_train_ma_rewards_2 = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/6processes/20231005-153224/results/train_ma_rewards.npy")
A3C_train_ma_rewards_2.tolist()

A3C_train_rewards_3 = np.load("C:/Users/23928/Desktop/结果/0.00002/20231124-103733/results/train_rewards.npy")
A3C_train_rewards_3.tolist()
A3C_train_ma_rewards_3 = np.load("C:/Users/23928/Desktop/结果/0.00002/20231124-103733/results/train_ma_rewards.npy")
A3C_train_ma_rewards_3.tolist()

plot_A3C_rewards_lr(A3C_train_ma_rewards_1,A3C_train_ma_rewards_2,A3C_train_ma_rewards_3)

#绘制绘制不同学习率A3C实验完成率收敛图
A3C_train_completion_rate_1 = np.load("C:/Users/23928/Desktop/结果/0.002/20231124-103556/results/train_completion_rate.npy")
A3C_train_completion_rate_1.tolist()
A3C_train_ma_completion_rate_1 = np.load("C:/Users/23928/Desktop/结果/0.002/20231124-103556/results/train_ma_completion_rate.npy")
A3C_train_ma_completion_rate_1.tolist()


A3C_train_completion_rate_2 = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/5processes/20231007-083546/results/train_completion_rate.npy")
A3C_train_completion_rate_2.tolist()
A3C_train_ma_completion_rate_2 = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/5processes/20231007-083546/results/train_ma_completion_rate.npy")
A3C_train_ma_completion_rate_2.tolist()

A3C_train_completion_rate_3 = np.load("C:/Users/23928/Desktop/结果/0.00002/20231124-103733/results/train_completion_rate.npy")
A3C_train_completion_rate_3.tolist()
A3C_train_ma_completion_rate_3 = np.load("C:/Users/23928/Desktop/结果/0.00002/20231124-103733/results/train_ma_completion_rate.npy")
A3C_train_ma_completion_rate_3.tolist()

plot_A3C_completion_rate_lr(A3C_train_ma_completion_rate_1,A3C_train_ma_completion_rate_2,A3C_train_ma_completion_rate_3)


#绘制对比实验奖励收敛图
#
A3C_train_rewards = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/6processes/20231005-153224/results/train_rewards.npy")
A3C_train_rewards.tolist()
A3C_train_ma_rewards = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/6processes/20231005-153224/results/train_ma_rewards.npy")
A3C_train_ma_rewards.tolist()

Greedy_train_rewards = np.load("C:/Users/23928/Desktop/结果/参数2/greedy/20231009-092855/results/train_rewards.npy")
Greedy_train_rewards.tolist()
Greedy_train_ma_rewards = np.load("C:/Users/23928/Desktop/结果/参数2/greedy/20231009-092855/results/train_ma_rewards.npy")
Greedy_train_ma_rewards.tolist()

DQN_train_rewards = np.load("C:/Users/23928/Desktop/结果/参数2/DQN/20231013-215609/results/train_rewards.npy")
DQN_train_rewards.tolist()
DQN_train_ma_rewards = np.load("C:/Users/23928/Desktop/结果/参数2/DQN/20231013-215609/results/train_ma_rewards.npy")
DQN_train_ma_rewards.tolist()

plot_contrast_rewards(A3C_train_ma_rewards,Greedy_train_ma_rewards,DQN_train_ma_rewards)

#绘制对比实验完成率收敛图

A3C_train_completion_rate = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/6processes/20231005-153224/results/train_completion_rate.npy")
A3C_train_completion_rate.tolist()
A3C_train_ma_completion_rate = np.load("C:/Users/23928/Desktop/结果/参数2/A3C/6processes/20231005-153224/results/train_ma_completion_rate.npy")
A3C_train_ma_completion_rate.tolist()

Greedy_train_completion_rate = np.load("C:/Users/23928/Desktop/结果/参数2/greedy/20231009-092855/results/train_completion_rate.npy")
Greedy_train_completion_rate.tolist()
Greedy_train_ma_completion_rate = np.load("C:/Users/23928/Desktop/结果/参数2/greedy/20231009-092855/results/train_ma_completion_rate.npy")
Greedy_train_ma_completion_rate.tolist()


DQN_train_completion_rate = np.load("C:/Users/23928/Desktop/结果/参数2/DQN/20231013-215609/results/train_completion_rate.npy")
DQN_train_completion_rate.tolist()
DQN_train_ma_completion_rate = np.load("C:/Users/23928/Desktop/结果/参数2/DQN/20231013-215609/results/train_ma_completion_rate.npy")
DQN_train_ma_completion_rate.tolist()


plot_contrast_completion_rate(A3C_train_ma_completion_rate,Greedy_train_ma_completion_rate,DQN_train_ma_completion_rate)

#绘制任务大小对平均延迟的影响图

A3C_train_rewards_2M = np.load("C:/Users/23928/Desktop/结果/参数2/tasksize/2/A3C/20231013-170138/results/train_rewards.npy")
A3C_train_rewards_2M.tolist()
A3C_train_rewards_2M=A3C_train_rewards_2M[:300]
A3C_average_2M = sum(A3C_train_rewards_2M) / len(A3C_train_rewards_2M)
A3C_train_rewards_3M = np.load("C:/Users/23928/Desktop/结果/参数2/tasksize/3/A3C/20231013-145512/results/train_rewards.npy")
A3C_train_rewards_3M.tolist()
A3C_train_rewards_3M=A3C_train_rewards_3M[:300]
A3C_average_3M = sum(A3C_train_rewards_3M) / len(A3C_train_rewards_3M)
A3C_train_rewards_4M = np.load("C:/Users/23928/Desktop/结果/参数2/tasksize/4/A3C/20231013-145114/results/train_rewards.npy")
A3C_train_rewards_4M.tolist()
A3C_train_rewards_4M=A3C_train_rewards_4M[:300]
A3C_average_4M = sum(A3C_train_rewards_4M) / len(A3C_train_rewards_4M)
A3C_train_rewards_5M = np.load("C:/Users/23928/Desktop/结果/参数2/tasksize/5/A3C/20231013-203310/results/train_rewards.npy")
A3C_train_rewards_5M.tolist()
A3C_train_rewards_5M=A3C_train_rewards_5M[:300]
A3C_average_5M = sum(A3C_train_rewards_5M) / len(A3C_train_rewards_5M)
A3C_train_rewards_6M = np.load("C:/Users/23928/Desktop/结果/参数2/tasksize/6/A3C/20231013-145236/results/train_rewards.npy")
A3C_train_rewards_6M.tolist()
A3C_train_rewards_6M=A3C_train_rewards_6M[:300]
A3C_average_6M = sum(A3C_train_rewards_6M) / len(A3C_train_rewards_6M)
A3C_average=[A3C_average_2M,A3C_average_3M,A3C_average_4M,A3C_average_5M,A3C_average_6M]

Greedy_train_rewards_2M = np.load("C:/Users/23928/Desktop/结果/参数2/tasksize/2/greedy/20231016-193257/results/train_rewards.npy")
Greedy_train_rewards_2M.tolist()
Greedy_train_rewards_2M=Greedy_train_rewards_2M[:300]
Greedy_average_2M = sum(Greedy_train_rewards_2M) / len(Greedy_train_rewards_2M)
Greedy_train_rewards_3M = np.load("C:/Users/23928/Desktop/结果/参数2/tasksize/3/greedy/20231013-185429/results/train_rewards.npy")
Greedy_train_rewards_3M.tolist()
Greedy_train_rewards_3M=Greedy_train_rewards_3M[:300]
Greedy_average_3M = sum(Greedy_train_rewards_3M) / len(Greedy_train_rewards_3M)
Greedy_train_rewards_4M = np.load("C:/Users/23928/Desktop/结果/参数2/tasksize/4/greedy/20231013-185412/results/train_rewards.npy")
Greedy_train_rewards_4M.tolist()
Greedy_train_rewards_4M=Greedy_train_rewards_4M[:300]
Greedy_average_4M = sum(Greedy_train_rewards_4M) / len(Greedy_train_rewards_4M)
Greedy_train_rewards_5M = np.load("C:/Users/23928/Desktop/结果/参数2/tasksize/5/greedy/20231018-101028/results/train_rewards.npy")
Greedy_train_rewards_5M.tolist()
Greedy_train_rewards_5M=Greedy_train_rewards_5M[:300]
Greedy_average_5M = sum(Greedy_train_rewards_5M) / len(Greedy_train_rewards_5M)
Greedy_train_rewards_6M = np.load("C:/Users/23928/Desktop/结果/参数2/tasksize/6/greedy/20231013-185457/results/train_rewards.npy")
Greedy_train_rewards_6M.tolist()
Greedy_train_rewards_6M=Greedy_train_rewards_6M[:300]
Greedy_average_6M = sum(Greedy_train_rewards_6M) / len(Greedy_train_rewards_6M)
Greedy_average=[Greedy_average_2M,Greedy_average_3M,Greedy_average_4M,Greedy_average_5M,Greedy_average_6M]


DQN_train_rewards_2M = np.load("C:/Users/23928/Desktop/结果/参数2/tasksize/2/DQN/20231018-102922/results/train_rewards.npy")
DQN_train_rewards_2M.tolist()
DQN_train_rewards_2M=DQN_train_rewards_2M[:300]
DQN_average_2M = sum(DQN_train_rewards_2M) / len(DQN_train_rewards_2M)
DQN_train_rewards_3M = np.load("C:/Users/23928/Desktop/结果/参数2/tasksize/3/DQN/20231021-091109/results/train_rewards.npy")
DQN_train_rewards_3M.tolist()
DQN_train_rewards_3M=DQN_train_rewards_3M[:300]
DQN_average_3M = sum(DQN_train_rewards_3M) / len(DQN_train_rewards_3M)
DQN_train_rewards_4M = np.load("C:/Users/23928/Desktop/结果/参数2/tasksize/4/DQN/20231024-140014/results/train_rewards.npy")
DQN_train_rewards_4M.tolist()
DQN_train_rewards_4M=DQN_train_rewards_4M[:300]
DQN_average_4M = sum(DQN_train_rewards_4M) / len(DQN_train_rewards_4M)
DQN_train_rewards_5M = np.load("C:/Users/23928/Desktop/结果/参数2/tasksize/5/DQN/20231017-084148/results/train_rewards.npy")
DQN_train_rewards_5M.tolist()
DQN_train_rewards_5M=DQN_train_rewards_5M[:300]
DQN_average_5M = sum(DQN_train_rewards_5M) / len(DQN_train_rewards_5M)
DQN_train_rewards_6M = np.load("C:/Users/23928/Desktop/结果/参数2/tasksize/6/DQN/20231018-101052/results/train_rewards.npy")
DQN_train_rewards_6M.tolist()
DQN_train_rewards_6M=DQN_train_rewards_6M[:300]
DQN_average_6M = sum(DQN_train_rewards_6M) / len(DQN_train_rewards_6M)
DQN_average=[DQN_average_2M,DQN_average_3M,DQN_average_4M,DQN_average_5M,DQN_average_6M]

plot_different_tasksize_average_rewards(A3C_average,Greedy_average,DQN_average)

#
# #绘制车辆数目对平均延迟的影响图
A3C_train_rewards_8 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle number/8/A3C/20231021-085750/results/train_rewards.npy")
A3C_train_rewards_8.tolist()
A3C_train_rewards_8=A3C_train_rewards_8[:300]
A3C_average_8 = sum(A3C_train_rewards_8) / len(A3C_train_rewards_8)
A3C_train_rewards_9 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle number/9/A3C/20231025-182720/results/train_rewards.npy")
A3C_train_rewards_9.tolist()
A3C_train_rewards_9=A3C_train_rewards_9[:300]
A3C_average_9 =sum(A3C_train_rewards_9) / len(A3C_train_rewards_9)
A3C_train_rewards_10 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle number/10/A3C/20231005-153224/results/train_rewards.npy")
A3C_train_rewards_10.tolist()
A3C_train_rewards_10=A3C_train_rewards_10[:300]
A3C_average_10 = sum(A3C_train_rewards_10) / len(A3C_train_rewards_10)
A3C_average_vehicle_number=[A3C_average_8,A3C_average_9,A3C_average_10]
A3C_average_vehicle_number_1=[-A3C_average_8,-A3C_average_9,-A3C_average_10]

Greedy_train_rewards_8 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle number/8/Greedy/20231108-091912/results/train_rewards.npy")
Greedy_train_rewards_8.tolist()
Greedy_train_rewards_8=Greedy_train_rewards_8[:300]
Greedy_average_8 = sum(Greedy_train_rewards_8) / len(Greedy_train_rewards_8)
Greedy_train_rewards_9 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle number/9/Greedy/20231108-102015/results/train_rewards.npy")
Greedy_train_rewards_9.tolist()
Greedy_train_rewards_9=Greedy_train_rewards_9[:300]
Greedy_average_9 = sum(Greedy_train_rewards_9) / len(Greedy_train_rewards_9)
Greedy_train_rewards_10 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle number/10/Greedy/20231009-092855/results/train_rewards.npy")
Greedy_train_rewards_10.tolist()
Greedy_train_rewards_10=Greedy_train_rewards_10[:300]
Greedy_average_10 = sum(Greedy_train_rewards_10) / len(Greedy_train_rewards_10)
Greedy_average_vehicle_number=[Greedy_average_8,Greedy_average_9,Greedy_average_10]
Greedy_average_vehicle_number_1=[-Greedy_average_8,-Greedy_average_9,-Greedy_average_10]


DQN_train_rewards_8 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle number/8/DQN/20231026-215121/results/train_rewards.npy")
DQN_train_rewards_8.tolist()
DQN_train_rewards_8=DQN_train_rewards_8[:300]
DQN_average_8 = sum(DQN_train_rewards_8) / len(DQN_train_rewards_8)
DQN_train_rewards_9 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle number/9/DQN/20231026-085306/results/train_rewards.npy")
DQN_train_rewards_9.tolist()
DQN_train_rewards_9=DQN_train_rewards_9[:300]
DQN_average_9 =sum(DQN_train_rewards_9) / len(DQN_train_rewards_9)
DQN_train_rewards_10 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle number/10/DQN/20231013-215609/results/train_rewards.npy")
DQN_train_rewards_10.tolist()
DQN_train_rewards_10=DQN_train_rewards_10[:300]
DQN_average_10 = sum(DQN_train_rewards_10) / len(DQN_train_rewards_10)
DQN_average_vehicle_number=[DQN_average_8,DQN_average_9,DQN_average_10]
DQN_average_vehicle_number_1=[-DQN_average_8,-DQN_average_9,-DQN_average_10]


plot_different_vehicle_number_average_rewards(A3C_average_vehicle_number,Greedy_average_vehicle_number,DQN_average_vehicle_number)
plot_different_vehicle_number_average_rewards_1(A3C_average_vehicle_number_1,DQN_average_vehicle_number_1,Greedy_average_vehicle_number_1)



# #绘制任务计算强度对平均延迟的影响图

A3C_train_rewards_300 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/300/A3C/20231005-153224/results/train_rewards.npy")
A3C_train_rewards_300.tolist()
A3C_train_rewards_300=A3C_train_rewards_300[:300]
A3C_average_300 = sum(A3C_train_rewards_300) / len(A3C_train_rewards_300)
A3C_train_rewards_325 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/325/A3C/20231118-143434/results/train_rewards.npy")
A3C_train_rewards_325.tolist()
A3C_train_rewards_325=A3C_train_rewards_325[:300]
A3C_average_325 = sum(A3C_train_rewards_325) / len(A3C_train_rewards_325)
A3C_train_rewards_350 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/350/A3C/20231125-100807/results/train_rewards.npy")
A3C_train_rewards_350.tolist()
A3C_train_rewards_350=A3C_train_rewards_350[:300]
A3C_average_350 = sum(A3C_train_rewards_350) / len(A3C_train_rewards_350)
A3C_train_rewards_375 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/375/A3C/20231123-164321/results/train_rewards.npy")
A3C_train_rewards_375.tolist()
A3C_train_rewards_375=A3C_train_rewards_375[:300]
A3C_average_375 = sum(A3C_train_rewards_375) / len(A3C_train_rewards_375)
A3C_train_rewards_400 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/400/A3C/20231123-085643/results/train_rewards.npy")
A3C_train_rewards_400.tolist()
A3C_train_rewards_400=A3C_train_rewards_400[:300]
A3C_average_400 = sum(A3C_train_rewards_400) / len(A3C_train_rewards_400)

# A3C_average=[A3C_average_300,A3C_average_325,A3C_average_350,A3C_average_375]
A3C_average=[A3C_average_300,A3C_average_325,A3C_average_350,A3C_average_375,A3C_average_400]
A3C_average_1=[-A3C_average_300,-A3C_average_325,-A3C_average_350,-A3C_average_375]

Greedy_train_rewards_300 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/300/Greedy/20231009-092855/results/train_rewards.npy")
Greedy_train_rewards_300.tolist()
Greedy_train_rewards_300=Greedy_train_rewards_300[:300]
Greedy_average_300 = sum(Greedy_train_rewards_300) / len(Greedy_train_rewards_300)
Greedy_train_rewards_325 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/325/Greedy/20231119-115112/results/train_rewards.npy")
Greedy_train_rewards_325.tolist()
Greedy_train_rewards_325=Greedy_train_rewards_325[:300]
Greedy_average_325 = sum(Greedy_train_rewards_325) / len(Greedy_train_rewards_325)
Greedy_train_rewards_350 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/350/Greedy/20231115-085643/results/train_rewards.npy")
Greedy_train_rewards_350.tolist()
Greedy_train_rewards_350=Greedy_train_rewards_350[:300]
Greedy_average_350 = sum(Greedy_train_rewards_350) / len(Greedy_train_rewards_350)
Greedy_train_rewards_375= np.load("C:/Users/23928/Desktop/结果/参数2/task computation/375/Greedy/20231118-143354/results/train_rewards.npy")
Greedy_train_rewards_375.tolist()
Greedy_train_rewards_375=Greedy_train_rewards_375[:300]
Greedy_average_375 = sum(Greedy_train_rewards_375) / len(Greedy_train_rewards_375)
Greedy_train_rewards_400 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/400/Greedy/20231114-161340/results/train_rewards.npy")
Greedy_train_rewards_400.tolist()
Greedy_train_rewards_400=Greedy_train_rewards_400[:300]
Greedy_average_400 = sum(Greedy_train_rewards_400) / len(Greedy_train_rewards_400)

# Greedy_average=[Greedy_average_300,Greedy_average_325,Greedy_average_350,Greedy_average_375]
Greedy_average=[Greedy_average_300,Greedy_average_325,Greedy_average_350,Greedy_average_375,Greedy_average_400]
Greedy_average_1=[-Greedy_average_300,-Greedy_average_325,-Greedy_average_350,-Greedy_average_375]

DQN_train_rewards_300 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/300/DQN/20231013-215609/results/train_rewards.npy")
DQN_train_rewards_300.tolist()
DQN_train_rewards_300=DQN_train_rewards_300[:300]
DQN_average_300 = sum(DQN_train_rewards_300) / len(DQN_train_rewards_300)
DQN_train_rewards_325 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/325/DQN/20231118-143432/results/train_rewards.npy")
DQN_train_rewards_325.tolist()
DQN_train_rewards_325=DQN_train_rewards_325[:300]
DQN_average_325 = sum(DQN_train_rewards_325) / len(DQN_train_rewards_325)
DQN_train_rewards_350 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/350/DQN/20231115-085643/results/train_rewards.npy")
DQN_train_rewards_350.tolist()
DQN_train_rewards_350=DQN_train_rewards_350[:300]
DQN_average_350 = sum(DQN_train_rewards_350) / len(DQN_train_rewards_350)
DQN_train_rewards_375 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/375/DQN/20231122-103710/results/train_rewards.npy")
DQN_train_rewards_375.tolist()
DQN_train_rewards_375=DQN_train_rewards_375[:300]
DQN_average_375 = sum(DQN_train_rewards_375) / len(DQN_train_rewards_375)
DQN_train_rewards_400 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/400/DQN/20231122-093638/results/train_rewards.npy")
DQN_train_rewards_400.tolist()
DQN_train_rewards_400=DQN_train_rewards_400[:300]
DQN_average_400 = sum(DQN_train_rewards_400) / len(DQN_train_rewards_400)

# DQN_average=[DQN_average_300,DQN_average_325,DQN_average_350,DQN_average_375]
DQN_average=[DQN_average_300,DQN_average_325,DQN_average_350,DQN_average_375,DQN_average_400]
DQN_average_1=[-DQN_average_300,-DQN_average_325,-DQN_average_350,-DQN_average_375]

print("A3C_task_computation_resource:",A3C_average)
print("Greedy_task_computation_resource:",Greedy_average)
print("DQN_task_computation_resource:",DQN_average)
plot_different_task_computation_resource_average_rewards(A3C_average,Greedy_average,DQN_average)
# plot_different_task_computation_resource_1(A3C_average_1,DQN_average_1,Greedy_average_1)



# 绘制车辆速度对平均延迟的影响图
A3C_train_rewards_20_25 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/20-25/A3C/20231123-092203/results/train_rewards.npy")
A3C_train_rewards_20_25.tolist()
A3C_train_rewards_20_25=A3C_train_rewards_20_25[:300]
A3C_average_20_25 = sum(A3C_train_rewards_20_25) / len(A3C_train_rewards_20_25)
A3C_train_rewards_25_30 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/25-30/A3C/20231018-092441/results/train_rewards.npy")
A3C_train_rewards_25_30.tolist()
A3C_train_rewards_25_30=A3C_train_rewards_25_30[:300]
A3C_average_25_30 = sum(A3C_train_rewards_25_30) / len(A3C_train_rewards_25_30)
A3C_train_rewards_30_35 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/30-35/A3C/20231123-181639/results/train_rewards.npy")
A3C_train_rewards_30_35.tolist()
A3C_train_rewards_30_35=A3C_train_rewards_30_35[:300]
A3C_average_30_35 = sum(A3C_train_rewards_30_35) / len(A3C_train_rewards_30_35)
A3C_train_rewards_35_40 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/35-40/A3C/20231123-161624/results/train_rewards.npy")
A3C_train_rewards_35_40.tolist()
A3C_train_rewards_35_40=A3C_train_rewards_35_40[:300]
A3C_average_35_40 = sum(A3C_train_rewards_35_40) / len(A3C_train_rewards_35_40)
A3C_average_speed=[A3C_average_20_25,A3C_average_25_30,A3C_average_30_35]
A3C_average_speed_1=[-A3C_average_20_25,-A3C_average_25_30,-A3C_average_30_35]

Greedy_train_rewards_20_25 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/20-25/Greedy/20231123-092221/results/train_rewards.npy")
Greedy_train_rewards_20_25.tolist()
Greedy_train_rewards_20_25=Greedy_train_rewards_20_25[:300]
Greedy_average_20_25 = sum(Greedy_train_rewards_20_25) / len(Greedy_train_rewards_20_25)
Greedy_train_rewards_25_30 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/25-30/Greedy/20231021-085551/results/train_rewards.npy")
Greedy_train_rewards_25_30.tolist()
Greedy_train_rewards_25_30=Greedy_train_rewards_25_30[:300]
Greedy_average_25_30 = sum(Greedy_train_rewards_25_30) / len(Greedy_train_rewards_25_30)
Greedy_train_rewards_30_35 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/30-35/Greedy/20231020-215837/results/train_rewards.npy")
Greedy_train_rewards_30_35.tolist()
Greedy_train_rewards_30_35=Greedy_train_rewards_30_35[:300]
Greedy_average_30_35 = sum(Greedy_train_rewards_30_35) / len(Greedy_train_rewards_30_35)
Greedy_train_rewards_35_40 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/35-40/Greedy/20231024-091846/results/train_rewards.npy")
Greedy_train_rewards_35_40.tolist()
Greedy_train_rewards_35_40=Greedy_train_rewards_35_40[:300]
Greedy_average_35_40 = sum(Greedy_train_rewards_35_40) / len(Greedy_train_rewards_35_40)
Greedy_average_speed=[Greedy_average_20_25,Greedy_average_25_30,Greedy_average_30_35]
Greedy_average_speed=[Greedy_average_20_25,Greedy_average_25_30,Greedy_average_30_35]
Greedy_average_speed_1=[-Greedy_average_20_25,-Greedy_average_25_30,-Greedy_average_30_35]

DQN_train_rewards_20_25 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/20-25/DQN/20231123-092211/results/train_rewards.npy")
DQN_train_rewards_20_25.tolist()
DQN_train_rewards_20_25=DQN_train_rewards_20_25[:300]
DQN_average_20_25 = sum(DQN_train_rewards_20_25) / len(DQN_train_rewards_20_25)
DQN_train_rewards_25_30 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/25-30/DQN/20231025-184522/results/train_rewards.npy")
DQN_train_rewards_25_30.tolist()
DQN_train_rewards_25_30=DQN_train_rewards_25_30[:300]
DQN_average_25_30 = sum(DQN_train_rewards_25_30) / len(DQN_train_rewards_25_30)
DQN_train_rewards_30_35= np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/30-35/DQN/20231123-181636/results/train_rewards.npy")
DQN_train_rewards_30_35.tolist()
DQN_train_rewards_30_35=DQN_train_rewards_30_35[:300]
DQN_average_30_35 = sum(DQN_train_rewards_30_35) / len(DQN_train_rewards_30_35)
DQN_train_rewards_35_40= np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/35-40/DQN/20231125-100807/results/train_rewards.npy")
DQN_train_rewards_35_40.tolist()
DQN_train_rewards_35_40=DQN_train_rewards_35_40[:300]
DQN_average_35_40 = sum(DQN_train_rewards_35_40) / len(DQN_train_rewards_35_40)
DQN_average_speed=[DQN_average_20_25,DQN_average_25_30,DQN_average_30_35]
DQN_average_speed_1=[-DQN_average_20_25,-DQN_average_25_30,-DQN_average_30_35]

print("A3C_average_speed:",A3C_average_speed)
print("Greedy_average_speed:",Greedy_average_speed)
print("DQN_average_speed:",DQN_average_speed)
plot_different_vehicle_speed_average_rewards(A3C_average_speed,Greedy_average_speed,DQN_average_speed)
plot_different_vehicle_speed_average_rewards_1(A3C_average_speed_1,DQN_average_speed_1,Greedy_average_speed_1)

# #绘制任务计算强度对平均延迟的影响图
#
# A3C_train_rewards_300 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/300/A3C/20231005-153224/results/train_rewards.npy")
# A3C_train_rewards_300.tolist()
# A3C_train_rewards_300=A3C_train_rewards_300[:250]
# A3C_average_300 = sum(A3C_train_rewards_300) / len(A3C_train_rewards_300)
# A3C_train_rewards_325 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/325/A3C/20231118-143434/results/train_rewards.npy")
# A3C_train_rewards_325.tolist()
# A3C_train_rewards_325=A3C_train_rewards_325[:250]
# A3C_average_325 = sum(A3C_train_rewards_325) / len(A3C_train_rewards_325)
# A3C_train_rewards_350 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/350/A3C/20231122-103057/results/train_rewards.npy")
# A3C_train_rewards_350.tolist()
# A3C_train_rewards_350=A3C_train_rewards_350[:250]
# A3C_average_350 = sum(A3C_train_rewards_350) / len(A3C_train_rewards_350)
# A3C_train_rewards_375 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/375/A3C/20231122-103437/results/train_rewards.npy")
# A3C_train_rewards_375.tolist()
# A3C_train_rewards_375=A3C_train_rewards_375[:250]
# A3C_average_375 = sum(A3C_train_rewards_375) / len(A3C_train_rewards_375)
# A3C_train_rewards_400 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/400/A3C/20231116-085110/results/train_rewards.npy")
# A3C_train_rewards_400.tolist()
# A3C_train_rewards_400=A3C_train_rewards_400[:250]
# A3C_average_400 = sum(A3C_train_rewards_400) / len(A3C_train_rewards_400)
#
# A3C_average=[A3C_average_300,A3C_average_325,A3C_average_350,A3C_average_375,A3C_average_400]
#
# Greedy_train_rewards_300 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/300/Greedy/20231009-092855/results/train_rewards.npy")
# Greedy_train_rewards_300.tolist()
# Greedy_train_rewards_300=Greedy_train_rewards_300[:250]
# Greedy_average_300 = sum(Greedy_train_rewards_300) / len(Greedy_train_rewards_300)
# Greedy_train_rewards_325 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/325/Greedy/20231119-115112/results/train_rewards.npy")
# Greedy_train_rewards_325.tolist()
# Greedy_train_rewards_325=Greedy_train_rewards_325[:250]
# Greedy_average_325 = sum(Greedy_train_rewards_325) / len(Greedy_train_rewards_325)
# Greedy_train_rewards_350 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/350/Greedy/20231115-085643/results/train_rewards.npy")
# Greedy_train_rewards_350.tolist()
# Greedy_train_rewards_350=Greedy_train_rewards_350[:250]
# Greedy_average_350 = sum(Greedy_train_rewards_350) / len(Greedy_train_rewards_350)
# Greedy_train_rewards_375= np.load("C:/Users/23928/Desktop/结果/参数2/task computation/375/Greedy/20231118-143354/results/train_rewards.npy")
# Greedy_train_rewards_375.tolist()
# Greedy_train_rewards_375=Greedy_train_rewards_375[:250]
# Greedy_average_375 = sum(Greedy_train_rewards_375) / len(Greedy_train_rewards_375)
# Greedy_train_rewards_400 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/400/Greedy/20231114-161340/results/train_rewards.npy")
# Greedy_train_rewards_400.tolist()
# Greedy_train_rewards_400=Greedy_train_rewards_400[:300]
# Greedy_average_400 = sum(Greedy_train_rewards_400) / len(Greedy_train_rewards_400)
#
# Greedy_average=[Greedy_average_300,Greedy_average_325,Greedy_average_350,Greedy_average_375,Greedy_average_400]
#
# DQN_train_rewards_300 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/300/DQN/20231013-215609/results/train_rewards.npy")
# DQN_train_rewards_300.tolist()
# DQN_train_rewards_300=DQN_train_rewards_300[:250]
# DQN_average_300 = sum(DQN_train_rewards_300) / len(DQN_train_rewards_300)
# DQN_train_rewards_325 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/325/DQN/20231118-143432/results/train_rewards.npy")
# DQN_train_rewards_325.tolist()
# DQN_train_rewards_325=DQN_train_rewards_325[:250]
# DQN_average_325 = sum(DQN_train_rewards_325) / len(DQN_train_rewards_325)
# DQN_train_rewards_350 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/350/DQN/20231115-085643/results/train_rewards.npy")
# DQN_train_rewards_350.tolist()
# DQN_train_rewards_350=DQN_train_rewards_350[:250]
# DQN_average_350 = sum(DQN_train_rewards_350) / len(DQN_train_rewards_350)
# DQN_train_rewards_375 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/375/DQN/20231122-103710/results/train_rewards.npy")
# DQN_train_rewards_375.tolist()
# DQN_train_rewards_375=DQN_train_rewards_375[:250]
# DQN_average_375 = sum(DQN_train_rewards_375) / len(DQN_train_rewards_375)
# DQN_train_rewards_400 = np.load("C:/Users/23928/Desktop/结果/参数2/task computation/400/DQN/20231116-085108/results/train_rewards.npy")
# DQN_train_rewards_400.tolist()
# DQN_train_rewards_400=DQN_train_rewards_400[:250]
# DQN_average_400 = sum(DQN_train_rewards_400) / len(DQN_train_rewards_400)
#
# DQN_average=[DQN_average_300,DQN_average_325,DQN_average_350,DQN_average_375,DQN_average_400]
#
# print("A3C_task_computation_resource:",A3C_average)
# print("Greedy_task_computation_resource:",Greedy_average)
# print("DQN_task_computation_resource:",DQN_average)
# plot_different_task_computation_resource_average_rewards(A3C_average,Greedy_average,DQN_average)
#
#
#
#
# # 绘制车辆速度对平均延迟的影响图
# A3C_train_rewards_20_25 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/20-25/A3C/20231123-092203/results/train_rewards.npy")
# A3C_train_rewards_20_25.tolist()
# A3C_train_rewards_20_25=A3C_train_rewards_20_25[:250]
# A3C_average_20_25 = sum(A3C_train_rewards_20_25) / len(A3C_train_rewards_20_25)
# A3C_train_rewards_25_30 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/25-30/A3C/20231018-092441/results/train_rewards.npy")
# A3C_train_rewards_25_30.tolist()
# A3C_train_rewards_25_30=A3C_train_rewards_25_30[:250]
# A3C_average_25_30 = sum(A3C_train_rewards_25_30) / len(A3C_train_rewards_25_30)
# A3C_train_rewards_30_35 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/30-35/A3C/20231025-084607/results/train_rewards.npy")
# A3C_train_rewards_30_35.tolist()
# A3C_train_rewards_30_35=A3C_train_rewards_30_35[:250]
# A3C_average_30_35 = sum(A3C_train_rewards_30_35) / len(A3C_train_rewards_30_35)
# A3C_train_rewards_35_40 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/35-40/A3C/20231028-182617/results/train_rewards.npy")
# A3C_train_rewards_35_40.tolist()
# A3C_train_rewards_35_40=A3C_train_rewards_35_40[:250]
# A3C_average_35_40 = sum(A3C_train_rewards_35_40) / len(A3C_train_rewards_35_40)
# A3C_average_speed=[A3C_average_20_25,A3C_average_25_30,A3C_average_30_35,A3C_average_35_40]
#
# Greedy_train_rewards_20_25 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/25-30/Greedy/20231021-085551/results/train_rewards.npy")
# Greedy_train_rewards_20_25.tolist()
# Greedy_train_rewards_20_25=Greedy_train_rewards_20_25[:250]
# Greedy_average_20_25 = sum(Greedy_train_rewards_20_25) / len(Greedy_train_rewards_20_25)
# Greedy_train_rewards_25_30 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/25-30/Greedy/20231021-085551/results/train_rewards.npy")
# Greedy_train_rewards_25_30.tolist()
# Greedy_train_rewards_25_30=Greedy_train_rewards_25_30[:250]
# Greedy_average_25_30 = sum(Greedy_train_rewards_25_30) / len(Greedy_train_rewards_25_30)
# Greedy_train_rewards_30_35 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/30-35/Greedy/20231020-215837/results/train_rewards.npy")
# Greedy_train_rewards_30_35.tolist()
# Greedy_train_rewards_30_35=Greedy_train_rewards_30_35[:250]
# Greedy_average_30_35 = sum(Greedy_train_rewards_30_35) / len(Greedy_train_rewards_30_35)
# Greedy_train_rewards_35_40 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/35-40/Greedy/20231024-091846/results/train_rewards.npy")
# Greedy_train_rewards_35_40.tolist()
# Greedy_train_rewards_35_40=Greedy_train_rewards_35_40[:250]
# Greedy_average_35_40 = sum(Greedy_train_rewards_35_40) / len(Greedy_train_rewards_35_40)
# Greedy_average_speed=[Greedy_average_20_25,Greedy_average_25_30,Greedy_average_30_35,Greedy_average_35_40]
#
# DQN_train_rewards_20_25 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/20-25/DQN/20231123-092211/results/train_rewards.npy")
# DQN_train_rewards_20_25.tolist()
# DQN_train_rewards_20_25=DQN_train_rewards_20_25[:250]
# DQN_average_20_25 = sum(DQN_train_rewards_20_25) / len(DQN_train_rewards_20_25)
# DQN_train_rewards_25_30 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/25-30/DQN/20231025-184522/results/train_rewards.npy")
# DQN_train_rewards_25_30.tolist()
# DQN_train_rewards_25_30=DQN_train_rewards_25_30[:250]
# DQN_average_25_30 = sum(DQN_train_rewards_25_30) / len(DQN_train_rewards_25_30)
# DQN_train_rewards_30_35= np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/30-35/DQN/20231021-091435/results/train_rewards.npy")
# DQN_train_rewards_30_35.tolist()
# DQN_train_rewards_30_35=DQN_train_rewards_30_35[:250]
# DQN_average_30_35 = sum(DQN_train_rewards_30_35) / len(DQN_train_rewards_30_35)
# DQN_train_rewards_35_40= np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/35-40/DQN/20231122-103850/results/train_rewards.npy")
# DQN_train_rewards_35_40.tolist()
# DQN_train_rewards_35_40=DQN_train_rewards_35_40[:250]
# DQN_average_35_40 = sum(DQN_train_rewards_35_40) / len(DQN_train_rewards_35_40)
# DQN_average_speed=[DQN_average_20_25,DQN_average_25_30,DQN_average_30_35,DQN_average_35_40]
#
# print("A3C_average_speed:",A3C_average_speed)
# print("Greedy_average_speed:",Greedy_average_speed)
# print("DQN_average_speed:",DQN_average_speed)
# plot_different_vehicle_speed_average_rewards(A3C_average_speed,Greedy_average_speed,DQN_average_speed)




# A3C_train_rewards_20_25 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/30-40/A3C/20231005-153224/results/train_rewards.npy")
# A3C_train_rewards_20_25.tolist()
# A3C_train_rewards_20_25=A3C_train_rewards_20_25[:250]
# A3C_average_20_25 = sum(A3C_train_rewards_20_25) / len(A3C_train_rewards_20_25)
# A3C_train_rewards_25_30 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/40-50/A3C/20231117-110445/results/train_rewards.npy")
# A3C_train_rewards_25_30.tolist()
# A3C_train_rewards_25_30=A3C_train_rewards_25_30[:250]
# A3C_average_25_30 = sum(A3C_train_rewards_25_30) / len(A3C_train_rewards_25_30)
# A3C_train_rewards_30_35 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/50-60/A3C/20231117-191857/results/train_rewards.npy")
# A3C_train_rewards_30_35.tolist()
# A3C_train_rewards_30_35=A3C_train_rewards_30_35[:250]
# A3C_average_30_35 = sum(A3C_train_rewards_30_35) / len(A3C_train_rewards_30_35)
# A3C_average_speed=[A3C_average_20_25,A3C_average_25_30,A3C_average_30_35]
#
# Greedy_train_rewards_20_25 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/30-40/Greedy/20231009-092855/results/train_rewards.npy")
# Greedy_train_rewards_20_25.tolist()
# Greedy_train_rewards_20_25=Greedy_train_rewards_20_25[:250]
# Greedy_average_20_25 = sum(Greedy_train_rewards_20_25) / len(Greedy_train_rewards_20_25)
# Greedy_train_rewards_25_30 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/40-50/Greedy/20231115-141809/results/train_rewards.npy")
# Greedy_train_rewards_25_30.tolist()
# Greedy_train_rewards_25_30=Greedy_train_rewards_25_30[:250]
# Greedy_average_25_30 = sum(Greedy_train_rewards_25_30) / len(Greedy_train_rewards_25_30)
# Greedy_train_rewards_30_35 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/50-60/Greedy/20231115-214151/results/train_rewards.npy")
# Greedy_train_rewards_30_35.tolist()
# Greedy_train_rewards_30_35=Greedy_train_rewards_30_35[:250]
# Greedy_average_30_35 = sum(Greedy_train_rewards_30_35) / len(Greedy_train_rewards_30_35)
# Greedy_average_speed=[Greedy_average_20_25,Greedy_average_25_30,Greedy_average_30_35]
#
# DQN_train_rewards_20_25 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/30-40/DQN/20231013-215609/results/train_rewards.npy")
# DQN_train_rewards_20_25.tolist()
# DQN_train_rewards_20_25=DQN_train_rewards_20_25[:250]
# DQN_average_20_25 = sum(DQN_train_rewards_20_25) / len(DQN_train_rewards_20_25)
# DQN_train_rewards_25_30 = np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/40-50/DQN/20231118-143630/results/train_rewards.npy")
# DQN_train_rewards_25_30.tolist()
# DQN_train_rewards_25_30=DQN_train_rewards_25_30[:250]
# DQN_average_25_30 = sum(DQN_train_rewards_25_30) / len(DQN_train_rewards_25_30)
# DQN_train_rewards_30_35= np.load("C:/Users/23928/Desktop/结果/参数2/vehicle speed/50-60/DQN/20231115-214152/results/train_rewards.npy")
# DQN_train_rewards_30_35.tolist()
# DQN_train_rewards_30_35=DQN_train_rewards_30_35[:250]
# DQN_average_30_35 = sum(DQN_train_rewards_30_35) / len(DQN_train_rewards_30_35)
# DQN_average_speed=[DQN_average_20_25,DQN_average_25_30,DQN_average_30_35]
# print("A3C_average_speed:",A3C_average_speed)
# print("Greedy_average_speed:",Greedy_average_speed)
# print("DQN_average_speed:",DQN_average_speed)