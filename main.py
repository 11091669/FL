import torch
import torch.nn as nn
from models.ResNet9 import ResNet9
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import sys
from worker import Worker
from multiprocessing import set_start_method,Manager
from ps import Ps
from test import test
import time
from data_preprocessing import get_cifar10, get_mnist
import os

if __name__ == '__main__':
    
    worker_nums_set = int(sys.argv[1])
    p_nums_set = int(sys.argv[2])
    data_mode = int(sys.argv[3])
    print('The p_num is ', p_nums_set)
    print('The worker_num is ', worker_nums_set)
    print('The data mode is ', data_mode)
    
    #分割数据集
    trainsets, testset = get_cifar10(worker_nums_set, data_mode)
    set_start_method('spawn')
    
    #创建进程
    man = Manager()
    reduceModel = man.list()                #放入规约好的模型
    for i in range(worker_nums_set):
        reduceModel.append(man.Queue())     #每个模型对应一个队列用来传输规约后模型
    update_times = man.Value('i', 0)        #更新次数
    
    workers = []
    # 初始化节点
    grads = man.list()                      #传输梯度
    for i in range(worker_nums_set):
        grads.append(man.Queue())           #每个模型对应一个队列用来传输梯度

    for worker_num in range(worker_nums_set):
        model = ResNet9(input_channels=3,output_channels= 10)
        dataset = trainsets[worker_num]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)
        workers.append(Worker(worker_num, grads[worker_num], reduceModel[worker_num], model, dataset, optimizer, scheduler)) 

    Pmodel = ResNet9(input_channels=3,output_channels= 10)
    optimizer = torch.optim.SGD(Pmodel.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)
    ps = Ps(Pmodel, optimizer, trainsets[0], p_nums_set, update_times, grads, reduceModel, worker_nums_set)

    #开始进程
    for i in range(worker_nums_set):
        workers[i].daemon = True
        workers[i].start()
    ps.daemon = True
    ps.start()

    #检查acc

    Tmodel = ResNet9(input_channels = 3,output_channels = 10)
    testloader = DataLoader(testset, batch_size = 128)
    reduceModel.append(None)

    #记录数组
    flag = 0
    record = [0,0,0]
    acc = 0
    while True :
        #设置收敛阈值
        if reduceModel[worker_nums_set] == None: continue
        if reduceModel[worker_nums_set] == -1: 
            for i in range(worker_nums_set):
                workers[i].terminate()
                workers[i].join()
            ps.terminate()
            ps.join()
            record[2] = int(acc)
            print(update_times.value)
            break
        Tmodel.load_state_dict(reduceModel[worker_nums_set])
        acc = test(Tmodel, testloader)
        time.sleep(1)
        if acc >= 80 and flag == 0:
            record[0] = update_times.value
            flag = 1
        elif acc >= 85 and flag == 1:
            record[1] = update_times.value
            flag = 2
        elif acc >= 90 :
            record[2] = update_times.value
            for i in range(worker_nums_set):
                workers[i].terminate()
                workers[i].join()
            ps.terminate()
            ps.join()
            break
    filename = 'record_cifar10_ResNet9.txt'
    with open(filename, 'a') as f:
        f.write('%d %d %d %d %d %d\n' %(worker_nums_set, p_nums_set, record[0], record[1], record[2], data_mode))
    f.close()
