# -*- coding: utf-8 -*-

import numpy as np
import singleFilePoccess
import math
import matplotlib.pyplot as plt
from pylab import *
# from tensorflow.python.keras.models import load_model
from keras.models import load_model
#import singleFilePoccess
import bigbatch
my_model = load_model('./resNet.h5')
ppppp=my_model.predict(np.zeros([1,32,32,2]))
def hangji():
    filep = './test_new/ScenarioFolder/'
    # fileq = './path/'
    fileq = './test_new/pathfolderlabel/'
    def testpath(environment, Spoint):
    #当前点无人机探测到的环境信息
        detectFile = singleFilePoccess.singleDetect(environment, Spoint)
    #神经网络测试输出
        label = testnerualnetwork(detectFile)
    #算出标签转化成偏转角度
    #theta1 = computeAngel(labelFile)
    #加上之前的角度
    #thetaSum = theta1 + thetaSum
    #算出更新点的位置
    #point = [labelFile[0][0]*100, labelFile[0][1]*100]
    #point = labelFile*100
        pl=np.zeros(8)
        step = 5
        if label == 0:
            point = [Spoint[0]+step, Spoint[1]]
            pl=[1,0,0,0,0,0,0,0]
        elif label == 1:
            point = [Spoint[0]+step, Spoint[1]+step]
            pl=[0,1,0,0,0,0,0,0]
        elif label == 2:
            point = [Spoint[0], Spoint[1]+step]
            pl=[0,0,1,0,0,0,0,0]
        elif label == 3:
            point = [Spoint[0]-step, Spoint[1]+step]
            pl=[0,0,0,1,0,0,0,0]
        elif label == 4:
            point = [Spoint[0]-step, Spoint[1]]
            pl=[0,0,0,0,1,0,0,0]
        elif label == 5:
            point = [Spoint[0]-step, Spoint[1]-step]
            pl=[0,0,0,0,0,1,0,0]
        elif label == 6:
            point = [Spoint[0], Spoint[1]-step]
            pl=[0,0,0,0,0,0,1,0]
        elif label == 7:
            point = [Spoint[0]+step, Spoint[1]-step]
            pl=[0,0,0,0,0,0,0,1]
        else:
            print(False)
    #point = labelFile
    #print(point)
    #返回下一点位置
        return point,pl
    
    def testnerualnetwork(testdata1):
    #扩维的
        x = np.expand_dims(testdata1, axis=0)
    #print(x.shape)
        testlabel1 = my_model.predict(x)
        maxaction = 0
        for i in range(len(testlabel1[0])):
            if testlabel1[0][i] > testlabel1[0][maxaction]:
                maxaction = i
    #[[0 1 0]]二维的array
        return maxaction
    
    sceneList, threatList = bigbatch.eachFile(filep)
    pathList = bigbatch.pathFile(fileq)
#print(len(pathList),len(threatList))
#if len(sceneList) == len(threatList):
   # if len(pathList) == len(threatList):
    size = len(pathList)
        #print('True')
    #else:
       # print('False')
#else:
    #print('False')
    for n in range(size):
        scene = np.loadtxt(sceneList[n]) # 读取场景信息，包括无人机探测到的环境信息，无人机位置，目标位置，无人机探测到的威胁信息，威胁位置，威胁类型，威胁速度，威胁方向，威胁半径，威胁强度，威胁持续时间
        threat = np.loadtxt(threatList[n])
        path = np.loadtxt(pathList[n])
        environment = singleFilePoccess.mergeEnvironment(scene, threat)
        pathPoint= path
        if pathPoint.shape[0]==2:
            continue
        startpoint = pathPoint[0]
        print(startpoint)
        endpoint = pathPoint[pathPoint.shape[0]-1]
        s=np.zeros(2)
        d=np.zeros(1)
        final=np.zeros(1)
        long=np.zeros(1)
        a = pathPoint.shape[0]+10      
        point2 = np.copy(startpoint)
        point1 = np.expand_dims(point2, axis=0)
        pointpath = np.copy(point1)
        pray=np.zeros((pathPoint.shape[0],8))
        while(a >= 1):
            point2 = point1[0]
            print('--------------------------')
            if np.sqrt(np.sum(np.square(point2-endpoint))) > 10:
                if point2[0]>=0 and point2[0]<=100 and point2[1]>=0 and point2[1]<=100:
                    point1[0],pray[pathPoint.shape[0]-a][:]= testpath(environment, point2)
                    print(point1[0])
                    if point1[0][0]>=0 and point1[0][0]<=100 and point1[0][1]>=0 and point1[0][1]<=100:
                        pointpath = np.vstack((pointpath, point1[0]))
                        #point1 = np.expand_dims(startpoint, axis=0)
                        print(pointpath)
                        a = a - 1
                    else:
                        a = 0
                        print("END")
                        print(n)
                else:
                    a = 0
                    print("END")
                    print(n)
            else:
                pointpath = np.vstack((pointpath, endpoint))
                a = 0
                print("END")
                print(n)
        if pointpath.shape[0]==2:
            continue
        for m in range (pointpath.shape[0]):
            if pointpath[m][0]>=0 and pointpath[m][0]<=100 and pointpath[m][1]>=0 and pointpath[m][1]<=100:
                s[1]=threat[int(pointpath[m][0])][int(pointpath[m][1])]+s[1]
            else:
                s[1]=s[1]+0
        for m in range (pathPoint.shape[0]):
            if pathPoint[m][0]>=0 and pathPoint[m][0]<=100 and pathPoint[m][1]>=0 and pathPoint[m][1]<=100:
                s[0]=threat[int(pathPoint[m][0])][int(pathPoint[m][1])]+s[0]
            else:
                s[0]=s[0]+0
        for m in range (2):
                d=d+np.sqrt(np.square(pathPoint[m+1][0]-pointpath[m+1][0])+np.square(pathPoint[m+1][1]-pointpath[m+1][1])+np.square(pathPoint[pathPoint.shape[0]-m-1][0]-pointpath[pointpath.shape[0]-m-1][0])+np.square(pathPoint[pathPoint.shape[0]-m-1][1]-pointpath[pointpath.shape[0]-m-1][1]))
        final=final+np.sqrt(np.square(pathPoint[pathPoint.shape[0]-1][1]-pointpath[pointpath.shape[0]-1][1])+np.square(pathPoint[pathPoint.shape[0]-1][0]-pointpath[pointpath.shape[0]-1][0]))
        long=long+pointpath.shape[0]-pathPoint.shape[0]
        print(d)
        print(s)
        d = np.concatenate((d, s))
        final = np.concatenate((final,d))
        long=np.concatenate((long,final))

        x = []
        y = []
        #plt.figure(figsize=(8,8))
        plt.figure(1)
        circleR = 16
        if int(startpoint[0] - circleR) > 0:
            x1 = int(startpoint[0] - circleR) 
        else:
            x1 = 0
        if int(startpoint[1] - circleR) > 0:
            y1 = int(startpoint[1] - circleR) 
        else:
            y1 = 0
        if int(startpoint[0] + circleR) <100:
            x2 = int(startpoint[0] + circleR) 
        else:
            x2 = 100
        if int(startpoint[1] + circleR) <100:
            y2 = int(startpoint[1] + circleR) 
        else:
            y2 = 100
        
        plt.axis([0, 100, 0, 100])
        for p in range(environment.shape[0]):
            for q in range(environment.shape[1]):
                if environment[p][q][1] > 0:
                    plt.scatter(p, q, c='b', alpha=environment[p][q][1])
                
        #num_pointpath = pointpath.shape
        #for i in range(num_pointpath[0]):
            #x.append(pointpath[i][0])
            #y.append(pointpath[i][1])
        plt.plot(pointpath[:,0], pointpath[:,1], "r-", marker='o', linewidth = 1)
        plt.plot(pathPoint[:,0], pathPoint[:,1], "g--", marker='o', linewidth = 1)
        if final[0]<=10:
            if s[0]<3:
                if s[1]<3:
                    if long[0]<=2:
                        if d[0]<=20:
                            plt.title('100')
                        else:
                            plt.title('85 not same')
                    else:
                        plt.title('80 not fast')
                else:
                    plt.title('60 not safe')
            elif s[1]<=s[0]:
                if long[0]<=2:
                     if d[0]<=20:
                         plt.title('100')
                     else:
                         plt.title('85 not same')
                else:
                    plt.title('80 not fast')
            else:
                plt.title('60 not safe')
        else:
            plt.title('fail')
        plt.savefig('./A/201902380002'+'%04d'%n+'.png')
        plt.close('all')


