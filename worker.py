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

#工作线程
class Worker(Process):
    def __init__(self, ID 
                 , queue
                 , reduce
                 , model
                 , dataset
                 , optimizer 
                 , scheduler = None
                 , criterion = nn.CrossEntropyLoss()):
        torch.cuda.set_device(ID % torch.cuda.device_count())
        super(Process,self).__init__()
        self._ID = ID
        self._model = model.cuda()
        self._queue = queue             #传递计算好的模型
        self._reduce = reduce           #接收规约后的模型
        self._dataset = dataset         #数据集
        self._dataloader = DataLoader(dataset, batch_size = 128, shuffle=True)
        self._optimizer = optimizer     #模型优化器
        self._scheduler = scheduler     #模型步长优化
        self._criterion = criterion     #损失函数


    def run(self):
        self.train(self._queue,self._reduce)

    # Training
    def train(self, queue, reduce):
        torch.cuda.set_device(self._ID % torch.cuda.device_count())
        
        for epoch in range(800):
            print('worker %d epoch %d' %(self._ID, epoch))
            #切换为训练模式
            self._model.cuda()
            self._model.train()
            for batch_idx, (inputs, targets) in enumerate(self._dataloader):
                #训练流程
                # inputs = inputs.view(-1, 28, 28)

                inputs = inputs.cuda()
                targets = targets.cuda()
                self._optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self._criterion(outputs, targets)
                gradients = torch.autograd.grad(loss, self._model.parameters())
                # self._optimizer.step() 不在更新模型，直接在ps端进行更新

                # 创建梯度的克隆副本
                cloned_gradients = [grad.clone().cpu() for grad in gradients]

                # 发送梯度副本到PS进行聚合
                queue.put(cloned_gradients)

                #阻塞等待信号
                reduce_model = reduce.get()

                # #将接收的模型加载到训练模型
                self._model.load_state_dict(reduce_model)
        
        # 超过800个epoch了
        queue.put(0)
