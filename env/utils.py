#!/usr/bin/env python
# coding=utf-8
"""
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 16:02:24
LastEditor: John
LastEditTime: 2022-07-13 22:15:46
Description:
Environment:
"""
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.font_manager import FontProperties  # 导入字体模块


def chinese_font():
    """ 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体 """
    try:
        font = FontProperties(
            fname='C:/Windows/Fonts/STSONG.TTF', size=15)  # fname系统字体路径，此处是windows的
    except:
        font = None
    return font






def plot_rewards_cn(rewards, ma_rewards, cfg, tag='train'):
    """ 中文画图 """
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(cfg.env_name, cfg.algo_name), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.legend((u'奖励', u'滑动平均奖励',), loc="best", prop=chinese_font())
    if cfg.save:
        plt.savefig(cfg.result_path + f"{tag}_rewards_curve_cn.eps", format='eps', dpi=1000)
    plt.show()


def plot_rewards(rewards, ma_rewards, cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {}".format(cfg.device, cfg.algo_name), fontsize=18)
    plt.xlabel('epsiodes', fontsize=18)
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    plt.grid()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_rewards_curve.eps".format(tag), format='eps', dpi=1000)
    plt.show()


def plot_completion_rate(completion_rate, ma_completion_rate, cfg, tag='train'):
    # sns.set()
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("learning curve on {} of {}".format(cfg.device, cfg.algo_name), fontsize=18)
    plt.xticks( fontsize=16, fontname='Times New Roman')
    plt.yticks( fontsize=16, fontname='Times New Roman')
    plt.xlabel('episodes', fontsize=18, fontname='Times New Roman')
    plt.ylabel('completion ratio', fontsize=18, fontname='Times New Roman')
    plt.plot(completion_rate, label='completion_rate')
    plt.plot(ma_completion_rate, label='ma_completion_rate')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size':18, 'family': 'Times New Roman'})
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_completion_rate_curve.eps".format(tag), format='eps', dpi=1000)
    plt.show()
########################################################################################################################
def plot_A3C_rewards(A3C_train_ma_rewards_1,A3C_train_ma_rewards_2,A3C_train_ma_rewards_3):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('rewards', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.plot(A3C_train_ma_rewards_1, label='number of training processes=2')
    plt.plot(A3C_train_ma_rewards_2, label='number of training processes=4')
    plt.plot(A3C_train_ma_rewards_3, label='number of training processes=6')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('A3C_rewards.pdf', format='pdf')
    plt.show()


def plot_A3C_rewards_lr(A3C_train_ma_rewards_1_lr,A3C_train_ma_rewards_2_lr,A3C_train_ma_rewards_3_lr):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('rewards', fontsize=26, fontname='Times New Roman')
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.plot(A3C_train_ma_rewards_1_lr, label='lr=0.002')
    plt.plot(A3C_train_ma_rewards_2_lr, label='lr=0.0002')
    plt.plot(A3C_train_ma_rewards_3_lr, label='lr=0.00002')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('A3C_rewards_lr.pdf', format='pdf')
    plt.show()


def plot_contrast_rewards(A3C_train_ma_rewards,Greedy_train_ma_rewards,DQN_train_ma_rewards):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    # plt.title("Convergence graphs of different algorithms", fontsize=14)
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')

    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('rewards', fontsize=26, fontname='Times New Roman')
    plt.plot(A3C_train_ma_rewards, label='A3C')
    plt.plot(Greedy_train_ma_rewards, label='Greedy')
    plt.plot(DQN_train_ma_rewards, label='DQN')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('contrast_rewards.pdf', format='pdf')
    plt.show()


def plot_A3C_completion_rate(A3C_train_ma_completion_rate_1,A3C_train_ma_completion_rate_2,A3C_train_ma_completion_rate_3):
    # sns.set()
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('completion ratio', fontsize=26, fontname='Times New Roman')
    plt.plot(A3C_train_ma_completion_rate_1, label='number of training processes=2')
    plt.plot(A3C_train_ma_completion_rate_2, label='number of training processes=4')
    plt.plot(A3C_train_ma_completion_rate_3, label='number of training processes=6')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('A3C_completion_rate.pdf', format='pdf')
    plt.show()



def plot_A3C_completion_rate_lr(A3C_train_ma_completion_rate_1_lr,A3C_train_ma_completion_rate_2_lr,A3C_train_ma_completion_rate_3_lr):
    # sns.set()
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("Convergence graph for different number of processess", fontsize=14)
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('completion ratio', fontsize=26, fontname='Times New Roman')
    plt.plot(A3C_train_ma_completion_rate_1_lr, label='lr=0.002')
    plt.plot(A3C_train_ma_completion_rate_2_lr, label='lr=0.0002')
    plt.plot(A3C_train_ma_completion_rate_3_lr, label='lr=0.00002')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('A3C_completion_rate_lr.pdf', format='pdf')
    plt.show()


def plot_contrast_completion_rate(A3C_train_ma_completion_rate,Greedy_train_ma_completion_rate,DQN_train_ma_completion_rate):
    # sns.set()
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("Convergence graphs of different algorithms", fontsize=14)
    plt.xticks( fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.xlabel('episodes', fontsize=26, fontname='Times New Roman')
    plt.ylabel('completion ratio', fontsize=26, fontname='Times New Roman')
    plt.plot(A3C_train_ma_completion_rate, label='A3C')
    plt.plot(Greedy_train_ma_completion_rate, label='Greedy')
    plt.plot(DQN_train_ma_completion_rate, label='DQN')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('contrast_completion_rate.pdf', format='pdf')
    plt.show()


def plot_different_tasksize_average_rewards(A3C_average,Greedy_average,DQN_average):
    # sns.set()
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("different tasksize average rewards", fontsize=18)
    plt.xlabel('size of tasks', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average rewards', fontsize=26, fontname='Times New Roman')
    x_datasize=[2,3,4,5,6]
    A3C_y_datasize=A3C_average
    Greedy_y_datasize= Greedy_average
    DQN_y_datasize=DQN_average
    plt.plot(x_datasize, A3C_y_datasize,label='A3C', linestyle='-', marker='o', markersize=8, markerfacecolor='red')
    plt.plot(x_datasize, Greedy_y_datasize,label='Greedy', linestyle='-', marker='s', markersize=8, markerfacecolor='orange')
    plt.plot(x_datasize, DQN_y_datasize,label='DQN', linestyle='-', marker='D', markersize=8, markerfacecolor='blue')
    plt.xticks(x_datasize,fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('different_tasksize_average_rewards.pdf', format='pdf')
    plt.show()




def plot_different_vehicle_number_average_rewards(A3C_average_vehicle_number,Greedy_average_vehicle_number,DQN_average_vehicle_number):
    # sns.set()
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("different vehicle number average rewards", fontsize=18)
    plt.xlabel('number of vehicles', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average rewards', fontsize=26, fontname='Times New Roman')
    x_datasize = [8, 9, 10]
    A3C_y_datasize = A3C_average_vehicle_number
    Greedy_y_datasize = Greedy_average_vehicle_number
    DQN_y_datasize = DQN_average_vehicle_number
    plt.plot(x_datasize, A3C_y_datasize, label='A3C', linestyle='-', marker='o', markersize=8, markerfacecolor='red')
    plt.plot(x_datasize, Greedy_y_datasize, label='Greedy', linestyle='-', marker='s', markersize=8,
             markerfacecolor='orange')
    plt.plot(x_datasize, DQN_y_datasize, label='DQN', linestyle='-', marker='D', markersize=8, markerfacecolor='blue')
    plt.xticks(x_datasize,fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('different_vehicle_number_average_rewards.pdf', format='pdf')
    plt.show()




def plot_different_vehicle_speed_average_rewards(A3C_speed,Greedy_speed,DQN_speed):
    # sns.set()
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("different vehicle speed average rewards", fontsize=18)
    plt.xlabel('speed of vehicles', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average rewards', fontsize=26, fontname='Times New Roman')
    x_datasize=['20-25','25-30','30-35']
    A3C_y_speed=A3C_speed
    Greedy_y_speed= Greedy_speed
    DQN_y_speed=DQN_speed
    plt.xticks(fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.plot(x_datasize, A3C_y_speed,label='A3C', linestyle='-', marker='o', markersize=8, markerfacecolor='red')
    plt.plot(x_datasize, Greedy_y_speed,label='Greedy', linestyle='-', marker='s', markersize=8, markerfacecolor='orange')
    plt.plot(x_datasize, DQN_y_speed,label='DQN', linestyle='-', marker='D', markersize=8, markerfacecolor='blue')
    plt.xticks(x_datasize,fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('different_vehicle_speed_average_rewards.pdf', format='pdf')
    plt.show()


def plot_different_task_computation_resource_average_rewards(A3C_task_computation_resource,Greedy_task_computation_resource,DQN_task_computation_resource):
    # sns.set()
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("different vehicle speed average rewards", fontsize=18)
    plt.xlabel('computation intensity of tasks', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average rewards', fontsize=26, fontname='Times New Roman')
    x_datasize=['300','325','350','375','400']
    A3C_y_task_computation_resource=A3C_task_computation_resource
    Greedy_y_task_computation_resource= Greedy_task_computation_resource
    DQN_y_task_computation_resource=DQN_task_computation_resource
    plt.xticks(fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.plot(x_datasize, A3C_y_task_computation_resource,label='A3C', linestyle='-', marker='o', markersize=8, markerfacecolor='red')
    plt.plot(x_datasize, Greedy_y_task_computation_resource,label='Greedy', linestyle='-', marker='s', markersize=8, markerfacecolor='orange')
    plt.plot(x_datasize, DQN_y_task_computation_resource,label='DQN', linestyle='-', marker='D', markersize=8, markerfacecolor='blue')
    plt.xticks(x_datasize,fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('different_task_computation_resource_average_rewards.pdf', format='pdf')
    plt.show()


########################################################################################################################

def plot_different_vehicle_speed_average_rewards_1(A3C_average_vehicle_speed,DQN_average_vehicle_speed,Greedy_average_vehicle_speed):
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.xlabel('speed of vehicles', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average rewards', fontsize=26, fontname='Times New Roman')
    Width = 0.2
    x1 = np.arange(len(A3C_average_vehicle_speed))
    x2=[x + Width for x in x1]
    x3=[x + Width for x in x2]
    A3C_y=A3C_average_vehicle_speed
    Greedy_y = Greedy_average_vehicle_speed
    DQN_y = DQN_average_vehicle_speed
    plt.xticks(fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.bar(x1, A3C_y,label='A3C',width=Width,color='red')
    plt.bar(x2, DQN_y, label='DQN',width=Width,color='orange')
    plt.bar(x3, Greedy_y, label='Greedy',width=Width,color='green')
    plt.xticks([r + Width for r in range(len(A3C_average_vehicle_speed))], ['20-25','25-30','30-35'])
    # 获取 y 轴刻度标签
    ticks = plt.gca().get_yticks()
    # 在每个刻度前面加上负号
    tick_labels = ['-' + str(int(abs(t))) if t != 0 else '0' for t in ticks]
    # 设置 y 轴刻度标签
    plt.yticks(ticks, tick_labels)
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('different_vehicle_speed_average_rewards.pdf', format='pdf')
    plt.show()


def plot_different_vehicle_number_average_rewards_1(A3C_average_rewards_vehicle_number,DQN_average_rewards_vehicle_number,Greedy_average_rewards_vehicle_number):
    plt.rcParams['figure.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景颜色为白色
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.xlabel('number of vehicles', fontsize=26, fontname='Times New Roman')
    plt.ylabel('average rewards', fontsize=26, fontname='Times New Roman')
    Width = 0.2
    x1 = np.arange(len(A3C_average_rewards_vehicle_number))
    x2=[x + Width for x in x1]
    x3=[x + Width for x in x2]
    A3C_y=A3C_average_rewards_vehicle_number
    Greedy_y = Greedy_average_rewards_vehicle_number
    DQN_y = DQN_average_rewards_vehicle_number
    plt.xticks(fontsize=22, fontname='Times New Roman')
    plt.yticks( fontsize=22, fontname='Times New Roman')
    plt.bar(x1, A3C_y,label='A3C',width=Width,color='red')
    plt.bar(x2, DQN_y, label='DQN',width=Width,color='orange')
    plt.bar(x3, Greedy_y, label='Greedy',width=Width,color='green')
    plt.xticks([r + Width for r in range(len(A3C_average_rewards_vehicle_number))], ['8','9','10'])
    # 获取 y 轴刻度标签
    ticks = plt.gca().get_yticks()
    # 在每个刻度前面加上负号
    tick_labels = ['-' + str(int(abs(t))) if t != 0 else '0' for t in ticks]
    # 设置 y 轴刻度标签
    plt.yticks(ticks, tick_labels)
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig('different_vehicle_number_average_rewards.pdf', format='pdf')
    plt.show()



def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo), fontsize=18)
    plt.xlabel('epsiodes', fontsize=18)
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path + "losses_curve.eps", format='eps', dpi=1000)
    plt.show()


def save_results_1(dic, tag='train', path='./results'):
    """ 保存奖励 """
    for key, value in dic.items():
        np.save(path + '{}_{}.npy'.format(tag, key), value)
    print('Results saved！')


def save_results(rewards, ma_rewards, tag='train', path='./results'):
    """ 保存奖励 """
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('Result saved!')



def make_dir(*paths):
    """ 创建文件夹 """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def del_empty_dir(*paths):
    """ 删除目录下所有空文件夹 """
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))


def save_args(args):
    # save parameters
    argsDict = args.__dict__
    with open(args.result_path + 'params.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    print("Parameters saved!")



#
# def save_results_from_npy_to_txt(file_path, output_path):
#     """ 从.npy文件中提取数据并保存为文本文件 """
#     data = np.load(file_path)
#     with open(output_path, 'w') as file:
#         for item in data:
#             file.write(' '.join(str(x) for x in item) + '\n')
#     print('Results extracted from .npy file and saved to text file！')



# def plot_offloading_rate(offloading_rate_c, offloading_rate_r, offloading_rate_v, cfg, tag='train'):
#     sns.set()
#     plt.figure()  # 创建一个图形实例，方便同时多画几个图
#     plt.title("offloading rate curve on {} of {}".format(cfg.device, cfg.algo_name), fontsize=18)
#     plt.xlabel('epsiodes', fontsize=18)
#     plt.plot(offloading_rate_c, label='offloading_rate_c')
#     plt.plot(offloading_rate_r, label='offloading_rate_r')
#     plt.plot(offloading_rate_v, label='offloading_rate_v')
#     plt.legend()
#     if cfg.save_fig:
#         plt.savefig(cfg.result_path + "{}_offloading_rate_curve.eps".format(tag), format='eps', dpi=1000)
#     plt.show()



# def plot_finish_rate(finish_rate, ma_finish_rate, cfg, tag='train'):
#     sns.set()
#     plt.figure()  # 创建一个图形实例，方便同时多画几个图
#     plt.title("finish rate curve on {} of {}".format(cfg.device, cfg.algo_name), fontsize=18)
#     plt.xlabel('epsiodes', fontsize=18)
#     plt.plot(finish_rate, label='finish rate')
#     plt.plot(ma_finish_rate, label='ma finish rate')
#     plt.legend()
#     if cfg.save_fig:
#         plt.savefig(cfg.result_path + "{}_finish_rate_curve.eps".format(tag), format='eps', dpi=1000)
#     plt.show()



