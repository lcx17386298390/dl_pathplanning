
#路径规划数据预处理
import numpy as np

#拼接距离和威胁矩阵
def mergeEnvironment(scene, threat):
    sizescene = scene.shape
    #sizethreat = threat.shape
    scene/=71
    environment = np.zeros([sizescene[0],sizescene[1],2])
    for i in range(sizescene[0]):
        for j in range(sizescene[1]):
            environment[i][j] = [scene[j][i],threat[j][i]]
    #print(environment)
    return environment

#无人机在当前航路点探测信息
def singleDetect(environment, pathPoint1):
    circleR = 16
    detectFile = np.zeros([2*circleR, 2*circleR, 2])
    for i in range(2*circleR):
        for j in range(2*circleR):
            if int(pathPoint1[0] + i -circleR) >=0 and int(pathPoint1[0] + i -circleR) <=100 and int(pathPoint1[1] + j -circleR) >= 0 and int(pathPoint1[1] + j -circleR) <= 100:
                detectFile[i][j] = environment[int(pathPoint1[0] + i -circleR)][int(pathPoint1[1] + j -circleR)]
            else:
                detectFile[i][j] = np.array([1, 1])
    return detectFile

#无人机在当前场景下，整条路径的探测信息合集
def pathDetect(pathPoint, environment):
    num = pathPoint.shape
    #num[0]
    for i in range(num[0]):
        if i == 0:
            detectSum = singleDetect(environment, pathPoint[i])
        elif i == 1:
            d = singleDetect(environment, pathPoint[i])
            detectSum = np.vstack(([detectSum], [d]))
        elif i >= 2:
            d = singleDetect(environment, pathPoint[i])
            detectSum = np.concatenate((detectSum, [d]))
    return detectSum

#无人机路径信息分割    
def dividePath(path):
    a = path.reshape(-1,10)
    print(a.shape)
    pathPoint = a[:,:2]
    pathLabel = a[:,2:]
    #pathPointLabel = path[1:,:2]*0.01
    pathPointLabel = a[:,:2]
    return pathPoint, pathLabel, pathPointLabel

#def singleLabel(pathlabels):
#    labelFile = pathlabels
#    return labelFile

    
#主函数
def singleScene(scenePath, threatPath, pathPath):
    scene = np.loadtxt(scenePath)
    
    threat = np.loadtxt(threatPath)
    
    path = np.loadtxt(pathPath)
    environment = mergeEnvironment(scene, threat)
    pathPoint, labelMat, pathPointLabel = dividePath(path)
    detectMat = pathDetect(pathPoint, environment)       
    return detectMat, labelMat

#detectFile, labelFile = singleScene(scenePath, threatPath, pathPath)