import sys, os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
print(curr_path)
sys.path.append(parent_path)  # 添加路径到系统路径
# parent_path_1 = os.path.dirname(parent_path)
# sys.path.append(parent_path_1)

import torch
import argparse

from env.utils import plot_rewards,  save_args, plot_completion_rate
from env.utils import save_results_1, make_dir
from env import environment

import datetime
from env.config import VehicularEnvConfig
from methods.Greedy.greedy_tasksize import Greedy


def get_args():
    """ Hyperparameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name',default='greedy',type=str,help="name of algorithm")
    #算法名称：A2C
    parser.add_argument('--env_name',default='Multihop-V2V',type=str,help="name of environment")
    parser.add_argument('--train_eps', default=300, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=200, type=int, help="episodes of testing")
    parser.add_argument('--result_path',
                        default=curr_path + "/outputs/" + parser.parse_args().env_name + '/' + curr_time + '/results/')
    parser.add_argument('--model_path',  # path to save models
                        default=curr_path + "/outputs/" + parser.parse_args().env_name + '/' + curr_time + '/models/')
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU
    return args



def train(cfg, env, agent):
    """ Training """
    print('Start training!')
    print(f'Env:{cfg.env_name}, A{cfg.algo_name}, 设备：{cfg.device}')
    rewards_plot = []
    ma_rewards_plot = []
    offloading_vehicle_number_plot = []
    offloading_rsu_number_plot = []
    offloading_cloud_number_plot = []
    completion_rate_plot=[]
    ma_completion_rate_plot = []
    for n_epi in range(cfg.train_eps):
        rewards=0
        steps=0
        done = False
        offloading_vehicle_number = 0
        offloading_rsu_number = 0
        offloading_cloud_number = 0
        complete_number =0
        state,function= env.reset()
        while not done:
            action = agent.choose_action( state,function)

            next_state, reward, done,next_function,offloading_vehicle,offloading_rsu,offloading_cloud,complete = env.step(action,function)
            state = next_state
            function=next_function
            steps+=1
            rewards += reward
            offloading_vehicle_number+=offloading_vehicle
            offloading_rsu_number+=offloading_rsu
            offloading_cloud_number+=offloading_cloud
            complete_number+=complete
        rewards_plot.append(rewards)
        offloading_vehicle_number_plot.append(offloading_vehicle_number)
        offloading_rsu_number_plot.append(offloading_rsu_number)
        offloading_cloud_number_plot.append( offloading_cloud_number)
        completion_rate = complete_number / (VehicularEnvConfig().rsu_number * (VehicularEnvConfig().time_slot_end + 1))
        completion_rate_plot.append(completion_rate)
        print("#  episode :{}, steps : {}, rewards : {}, complete : {}, vehicle : {}, rsu : {}, cloud : {}"
              .format(n_epi+1,steps, rewards,
                      completion_rate,offloading_vehicle_number,offloading_rsu_number,offloading_cloud_number))
        # time.sleep(1)

        if ma_rewards_plot:
            ma_rewards_plot.append(0.9 * ma_rewards_plot[-1] + 0.1 * rewards)
        else:
            ma_rewards_plot.append(rewards)

        if ma_completion_rate_plot:
            ma_completion_rate_plot.append(0.9 * ma_completion_rate_plot[-1] + 0.1 * completion_rate)
        else:
            ma_completion_rate_plot.append(completion_rate)


    res_dic_rewards = {'rewards': rewards_plot, 'ma_rewards': ma_rewards_plot}
    res_dic_completion_rate = {'completion_rate': completion_rate_plot, 'ma_completion_rate': ma_completion_rate_plot}
    if not os.path.exists(cfg.result_path):
        os.makedirs(cfg.result_path)
    save_results_1(res_dic_rewards, tag='train',
                   path=cfg.result_path)
    save_results_1(res_dic_completion_rate, tag='train',
                   path=cfg.result_path)
    plot_rewards(res_dic_rewards['rewards'], res_dic_rewards['ma_rewards'], cfg, tag="train")
    plot_completion_rate(res_dic_completion_rate['completion_rate'], res_dic_completion_rate['ma_completion_rate'], cfg, tag="train")
    env.close()





if __name__ == "__main__":
    cfg = get_args()
    # 训练
    env = environment.RoadState()
    agent = Greedy()
    train(cfg, env, agent)



    #20000-250000
    #8