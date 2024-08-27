import os
import numpy as np

fileq = './pathfolder/'

def pathFile(filepath):
    #F：读路径文件名列表
    #
    pathDir = os.listdir(filepath)
    pathList = []
    for s in pathDir:
        newDir=os.path.join(filepath,s)
        pathList.append(newDir)
    return pathList    
    
def pre_path(file):
    pathList = pathFile(file)
    size0 = len(pathList)
    labelnum = 8
    for i in range(size0):
        path = np.loadtxt(pathList[i])
        end = path.shape[0]
        pathlabel = np.zeros([end - 1, 2 + labelnum])
        for j in range(end - 1):
            pathlabel[j][:2] = path[j]
            deltaX = path[j+1][0] - path[j][0]
            deltaY = path[j+1][1] - path[j][1]
            if abs(deltaX) < 1.5:
                if deltaY < 0:
                    pathlabel[j][2:] = np.array([0, 0, 0, 0, 0, 0, 1, 0,])
                else:
                    pathlabel[j][2:] = np.array([0, 0, 1, 0, 0, 0, 0, 0,])
            elif deltaX > 0:
                if deltaY < 0:
                    pathlabel[j][2:] = np.array([0, 0, 0, 0, 0, 0, 0, 1,])
                elif abs(deltaY) < 1.5:
                    pathlabel[j][2:] = np.array([1, 0, 0, 0, 0, 0, 0, 0,])
                else:
                    pathlabel[j][2:] = np.array([0, 1, 0, 0, 0, 0, 0, 0,])
            else:
                if deltaY < 0:
                    pathlabel[j][2:] = np.array([0, 0, 0, 0, 0, 1, 0, 0,])
                elif abs(deltaY) < 1.5:
                    pathlabel[j][2:] = np.array([0, 0, 0, 0, 1, 0, 0, 0,])
                else:
                    pathlabel[j][2:] = np.array([0, 0, 0, 1, 0, 0, 0, 0,])
            
            
        np.savetxt('./pathfolderlabel/2019032600'+ '%04d'%i +'.txt', pathlabel, fmt='%s', newline='\r\n')
    return

pre_path(fileq)
