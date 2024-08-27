# -*- coding: utf-8 -*-

import os
import numpy as np
import singleFilePoccess

filep = './test1/ScenarioFolder/'  #"测试集场景信息文件位置"
fileq = './test1/pathfolderlabel/'  #"测试集航路信息文件位置"
def eachFile(filepath):
    pathDir = os.listdir(filepath)      #获取当前路径下的文件名，返回List
    pathDir.sort(key=lambda x: int(x[:-5]))
    sceneList = []
    threatList = []
    for s in pathDir:
        newDir=os.path.join(filepath,s)
        #将文件命加入到当前文件路径后面
        #print(newDir)
        if os.path.isfile(newDir) :         #如果是文件
            if os.path.splitext(newDir)[1]==".txt":  #判断是否是txt
                #sceneFileName, threatFileName = readFile(newDir)
                if 'B' in newDir:
                    threatList.append(newDir)
                else:
                    sceneList.append(newDir)                              
        else:
            eachFile(newDir)                #如果不是文件，递归这个文件夹的路径
    print(len(sceneList), len(threatList))
    return sceneList, threatList 

def pathFile(filepath):
    pathDir = os.listdir(filepath)
    pathDir.sort(key=lambda x: int(x[:-4]))
    pathList = []
    for s in pathDir:
        newDir=os.path.join(filepath,s)
        pathList.append(newDir)
    print(len(pathList))
    return pathList
            
#def readFile(fileName):
#    scenefileName = []
#    threatfileName = []
#    if 'A' in fileName:
#        scenefileName = fileName
#    elif 'B' in fileName:
#        threatfileName = fileName
#    return scenefileName, threatfileName

def pre():
    sceneList, threatList = eachFile(filep)
    pathList = pathFile(fileq)
#print(len(pathList),len(threatList))
    if len(sceneList) == len(threatList):
        if len(pathList) == len(threatList):
            size = len(pathList)
            #print('True')
        else:
            print('False')
    else:
        print('False')
      
    for i in range(size):
        if i == 0:
            detectFile, labelFile = singleFilePoccess.singleScene(sceneList[i], threatList[i], pathList[i])

        else:
            detect, label = singleFilePoccess.singleScene(sceneList[i], threatList[i], pathList[i])
            if np.size(label,0) == 1:
                #detect=np.expand_dims(detect1, axis=0)
            #else:
                #detect=detect1
                i=i
            else:   
                print(detect.shape)
                print(label.shape)
                detectFile = np.concatenate((detectFile, detect))
                labelFile = np.concatenate((labelFile, label))
                print(i)
    return detectFile,labelFile

#pre()
    #print(detectFile)
    #print(labelFile)
    #print(detectFile.shape)
    #print(labelFile.shape)
