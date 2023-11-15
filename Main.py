
"""
20230504
1. 要实现 学习率逐渐下降
2. 梯度裁剪
3. 最大忍耐度
4. 展示实验结果
5. 写注释
6. 保存参数模型参数
7. debug ！！！！！！
8. gpu跑

20230519
1. 动态生成训练集
2.


model.eval() - model.eval()不会影响各层的gradient计算行为，即gradient计算和存储与training模式一样，只是不进行反向传播(backprobagation)
torch.no_grad() 用于停止autograd模块的工作，起到加速和节省显存的作用（具体行为就是停止gradient计算，从而节省了GPU算力和显存）
"""

import gc
import os
import pickle
import shutil
import time
import warnings

import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import cm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from codes.DATASET import DATASET
from codes.Function import classEmbeddingPath, textEmbeddingPath, classPrefix, saveImageArrayDir, textPrefix, \
    saveModelWightsDir, scoreNames, displayScore, preMain
from codes.Function import try_gpu, getScore, modelScoresVision
from codes.MyDataSet import MyDataSet
from codes.NNManager import Net

warnings.filterwarnings('ignore')

class Main:
    def __init__(self, device='cuda', isJoin=True):
        self.modelName = None
        self.lr = 1e-5  # 学习率
        self.nHidden = 256  # 隐藏层 - Bi-LSTM
        self.seqLen = 80  # 步长 - Bi-LSTM
        self.numLayers = 4  # 隐藏层层数
        self.batchSize = 4  # 批量
        self.maxClipping = 5  # 梯度裁剪
        self.normType = 2  # 梯度的范式
        self.dropout = 0.3  # DropOut层的概率 留取80%
        self.maxEpoch = 19  # 最大迭代 >= 3
        self.displayStep = 1  # 多少轮后展示训练结果ExtractFeature.py  =1时 会记录每个人epoch 当!=1时 记录maxEpoch//displayStep
        self.maxPatience = 25  # 能够容忍多少个epoch内都没有improvement 后期也不用了前期可调
        self.isJoin = isJoin
        self.representationScores = {}
        self.lrRecord = []  # 记录学习率变化
        self.scoreNames = scoreNames
        self.XExample = None  # 获得某一个X的样本
        self.device = device
        self.beforeEpoch = 0  # 可以继续训练
        self.net = Net(self.nHidden, self.seqLen, dropout=self.dropout, classEmbeddingPath=classEmbeddingPath,
                       textEmbeddingPath=textEmbeddingPath, device=device, numLayers=self.numLayers)
        self.net.apply(Net.weight_init)
        # self.net.load_state_dict(torch.load('outputs/2023-08-17/2023-08-17.pth'))  ### 加载预训练模型
        if device == 'cuda':
            self.net.to(device=try_gpu())
        self.loss = nn.BCELoss(reduction='none')
        self.updater = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.updater,
            T_0=5,  # 初试周期
            T_mult=2
        )
        self.trainIter = self.loadData(DATASET.TRAIN, isJoin=self.isJoin)
        self.testIter = self.loadData(DATASET.TEST, isJoin=self.isJoin)
        self.validIter = self.loadData(DATASET.VALID, isJoin=self.isJoin)
        self.logs = []

    def loadData(self, dataType=DATASET.TRAIN, isJoin=True):
        data = MyDataSet(
            seqLen=self.seqLen,
            imageClassDir=classPrefix,
            imageVectorDir=saveImageArrayDir,
            textDir=textPrefix,
            dataType=dataType,
            isJoin=isJoin
        )
        return DataLoader(dataset=data, batch_size=self.batchSize, shuffle=True, num_workers=0, drop_last=True)
        # pin_memory=True, prefetch_factor=2, persistent_workers=False)


############################# 混淆矩阵  ####################
    def confusion_matrix(self, preds, labels, conf_matrix):
        #preds = torch.argmax(preds, 1)
        for p, t in zip(preds, labels):
            conf_matrix[p.long(), t.long()] += 1
        return conf_matrix

############################# 混淆矩阵  ####################

    def conf_matrix_test(self, conf_matrix, dataType=DATASET.TEST, num=2000):

        if isinstance(self.net, torch.nn.Module):
            self.net.eval()
        with torch.no_grad():
            testData = self.validIter
            count = 0
            yPred, yTrue = [], []
            for X, y, _ in testData:
                self.XExample = X
                if self.device == "cuda":
                    y = y.cuda()
                y_pred = self.net(X)
                y_pred = y_pred.squeeze()
                y_pred = (y_pred > 0.5).type(torch.float32)
                ############################### 混淆矩阵 ##########
                conf_matrix = self.confusion_matrix(y_pred, y, conf_matrix)
                conf_matrix = conf_matrix.cpu()

                ###############################  混淆矩阵 ##########

                count += y_pred.shape[0]
                yPred.append(y_pred)
                yTrue.append(y)
        return conf_matrix

    def test(self, dataType=DATASET.TEST, num=2000):
        if isinstance(self.net, torch.nn.Module):
            self.net.eval()
        with torch.no_grad():
            if dataType == DATASET.TRAIN:
                testData = self.trainIter
            elif dataType == DATASET.TEST:
                testData = self.testIter
            else:
                testData = self.validIter
            count = 0
            yPred, yTrue = [], []
            for X, y, _ in testData:
                self.XExample = X
                if self.device == "cuda":
                    y = y.cuda()
                y_pred = self.net(X)
                count += y_pred.shape[0]
                if (dataType == DATASET.TRAIN) and (count > num):
                    break
                yPred.append(y_pred)
                yTrue.append(y)
        yPred = torch.cat(yPred, dim=0)
        yTrue = torch.cat(yTrue, dim=0)
        return getScore(y_pred=yPred.to(torch.device("cpu")), y_true=yTrue.to(torch.device("cpu")))
    
    def train_epoch(self):
        if isinstance(self.net, torch.nn.Module):
            self.net.train()
        if not isinstance(self.updater, torch.optim.Optimizer):
            raise AttributeError
        count = 0
        logs = tqdm(range(len(self.trainIter)))
        for X, i in zip(self.trainIter, logs):
            X, y, _ = X
            torch.cuda.empty_cache()  # 回收显存
            if self.device == "cuda":
                y = y.cuda()
            y_hat = self.net(X)
            self.lrRecord.append(self.updater.state_dict()['param_groups'][0]['lr'])
            l = self.loss(y_hat.squeeze(), y.squeeze()).mean()
            self.updater.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.maxClipping, norm_type=self.normType)
            self.updater.step()
            self.lr_scheduler.step()
            del X, y
            count += 1
            # if count >= 10:  # 调试用
            #     break
            gc.collect()  # 回收内存

    def train(self):
        maxF1 = 0  # 以F1score为指标
        patience = self.maxPatience  # 当前的容忍度
        start = time.time()
        confusion_matrix_list = []
        i = 0
        for epoch in range(self.beforeEpoch, self.beforeEpoch + self.maxEpoch):
            end = time.time()
            epochStr = "----epoch:{} total cost:{:.2f} min  ---------".format(epoch, (end - start) / 60)
            print(epochStr)
            self.train_epoch()
            if epoch % self.displayStep == 0:
                # acc, pre, rec, f1, auc, loss # 元组内的顺序
                trainScores, testScores, validScores = self.test(DATASET.TRAIN), self.test(DATASET.TEST), self.test(
                    DATASET.VALID)
                self.representationScores[epoch // self.displayStep] = tuple(
                    zip(trainScores, testScores, validScores))  #
                trainStr = "train, patience={}, acc:{:.3f}, pre:{:.3f}, rec:{:.3f}, f1:{:.3f}, acu:{:.3f}, loss:{:.2f}\n".format(
                    patience, *trainScores)
                testStr = "test, patience={}, acc:{:.3f}, pre:{:.3f}, rec:{:.3f}, f1:{:.3f}, acu:{:.3f}, loss:{:.2f}\n".format(
                    patience, *testScores)
                validStr = "valid, patience={}, acc:{:.3f}, pre:{:.3f}, rec:{:.3f}, f1:{:.3f}, acu:{:.3f} loss:{:.2f}\n".format(
                    patience, *validScores)
                self.logs.extend([epochStr + "\n", trainStr, testStr, validStr])
                print(trainStr, testStr, validStr, sep="")
                if validScores[0] > maxF1 + 1e-3:
                    maxF1, patience = validScores[3], self.maxPatience
                    ################################ 混淆矩阵 ###############
                    emotion_kinds = 2
                    confusion_matrix = torch.zeros(emotion_kinds, emotion_kinds)
                    conf_matrix = self.conf_matrix_test(confusion_matrix, DATASET.VALID, 2000)
                    conf_matrix = np.array(conf_matrix.cpu())
                    corrects = conf_matrix.diagonal(offset=0)
                    per_kinds = conf_matrix.sum(axis=1)
                    print("混淆矩阵总元素个数：{0}".format(int(np.sum(conf_matrix))))
                    print(conf_matrix)
                    print("每种情感总个数：", per_kinds)
                    print("每种情感预测正确的个数：", corrects)
                    print("每种情感的识别准确率为：{0}".format([rate*100 for rate in corrects/per_kinds]))
                    ###############################  混淆矩阵 ##############
                    confusion_matrix_list.append(conf_matrix)
                    i += 1
                    self.saveNet("bestModel", describe="bestModel")
                else:
                    patience -= 1
                #                 if patience == 0: # 是否打开
                #                     break
        self.saveNet()
        return confusion_matrix_list[i-1]
        
    def writeErrorSampleIds(self, dataType=DATASET.TEST):
        self.net.eval()
        with torch.no_grad():
            yPred, yTrue, ids = [], [], []
            if dataType == DATASET.TRAIN:
                testData = self.trainIter
            elif dataType == DATASET.TEST:
                testData = self.testIter
            else:
                testData = self.validIter
            for X, y, id in testData:
                yHat = self.net.forward(X).reshape(y.shape).to(torch.device("cpu"))
                yHat = (yHat >= 0.5).type(torch.int)
                y.to(torch.device(yHat.device))
                yPred.append(yHat)
                yTrue.append(y)
                ids.append(id)
        yPred = torch.cat(yPred, dim=0)
        yTrue = torch.cat(yTrue, dim=0)
        ids = numpy.array(ids).flatten()
        with open(saveModelWightsDir + self.modelName + "/Error.txt", "w+", encoding="utf-8") as file:
            [file.write(fileName + "\n") for fileName in ids[yPred != yTrue]]        

    def saveNet(self, saveName=time.strftime("%Y-%m-%d", time.localtime()), describe="unKnown"):
        """保存网络参数"""
        savePath = saveModelWightsDir + saveName + "/"
        self.modelName = saveName

        if os.path.exists(savePath):
            shutil.rmtree(savePath)  # 如果重新运行时，切忌如果有相同的文件名时要提前保存！！！！！
        os.makedirs(savePath, exist_ok=True)
        if not os.path.exists(savePath + "runs/"):
            os.mkdir(savePath + "runs/")

        torch.save(self.net.state_dict(), savePath + saveName + ".pth")

        summaryWriter = SummaryWriter(log_dir=savePath + "runs/")
        modelScoresVision(summaryWriter, scoresValues=self.representationScores, scoresNames=self.scoreNames,
                          lrValues=self.lrRecord)
        summaryWriter.close()

        runLogs = (self.representationScores, self.lrRecord)
        with open(savePath + "logs.py3", 'wb+') as f:
            pickle.dump(runLogs, f)

        with open(savePath + "describe.txt", 'w+') as f:
            f.write("acc, pre, rec, f1, auc, loss\n")
            f.writelines(self.logs)
        if describe != "bestModel":
            displayScore(saveName)

    def loadNet(self, loadName=time.strftime("%Y-%m-%d", time.localtime()), isEval=False):
        """加载网络参数"""
        loadPath = saveModelWightsDir + loadName + "/"
        self.modelName = loadName

        self.net.load_state_dict(torch.load(loadPath + loadName + ".pth"))

        with open(loadPath + "logs.py3", 'rb') as f:
            self.representationScores, self.lrRecord = pickle.load(f)
        self.beforeEpoch = len(self.representationScores)

        if isEval:
            self.net.eval()  # 不启用 BatchNormalization 和 Dropout


if __name__ == "__main__":
    print("<---------- 开始运行 祈祷不报错 ---------->")
    preMain(isSplit=False)  # 默认不进行数据增强，不改变数据分割
    main = Main(device='cuda')
    main.train()
    print("<---------- 运行结束 万幸 --------------->")
