import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader
from multiprocessing import Process, Queue
import time
import random
from copy import deepcopy

class Ps(Process):
    def __init__(self
                 , model
                 , optimizer
                 , data_set
                 , p_nums_set
                 , update_times
                 , q
                 , reduce_flag
                 , worker_nums_set 
                 , scheduler = None
                 , criterion = nn.CrossEntropyLoss()):
        super(Process,self).__init__()
        self._model = model
        self._queue = q           
        self._reduce = reduce_flag        
        self._dataset = data_set         #数据集
        self._dataloader = DataLoader(data_set, batch_size=256, shuffle=True)        
        self._update_times = update_times
        self._optimizer = optimizer     #模型优化器
        self._scheduler = scheduler     #模型步长优化
        self._criterion = criterion     #损失函数
        self._worker_nums_set = worker_nums_set
        self._p_nums_set = p_nums_set

    def run(self):
        self.train(self._queue,self._reduce)

    # Training
    def train(self, queue, reduce):
        torch.cuda.set_device(0 % torch.cuda.device_count())
        self._model.train()     
        while True :
            # outputs = self._model(inputs)
            self._optimizer.zero_grad()
            workers = []
            for i in range(self._worker_nums_set):
                m = queue[i].get()
                #有worker结束训练了
                if m == 0 :
                    reduce[self._worker_nums_set] = -1
                    return
                workers.append(m)
            # 随机选p个进行平均
            p_workers = random.sample(workers, self._p_nums_set)
            # 计算平均梯度
            average_gradients = []
            for params in zip(*p_workers):  # 按参数分组
                avg_grad = torch.stack(params).mean(dim=0)  # 计算平均值
                average_gradients.append(avg_grad)
            for param, avg_grad in zip(self._model.parameters(), average_gradients):
                param.grad = avg_grad  # 赋值平均梯度
            self._optimizer.step()   
            for i in range(self._worker_nums_set) :
                reduce[i].put(self._model.state_dict()) 
            if len(reduce) == self._worker_nums_set :
                reduce.append(self._model.state_dict())
            reduce[self._worker_nums_set] = self._model.state_dict()
            self._update_times.value += 1
            
    

