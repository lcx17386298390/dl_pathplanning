import os
import numpy as np


filep = './s/'
fileq = './p/'
def eachFile(filepath):
    #F：读出两类文件的题目列表
    #output：列表
    pathDir = os.listdir(filepath)      #获取当前路径下的文件名，返回List
    sceneList = []
    threatList = []
    for s in pathDir:
        newDir=os.path.join(filepath,s)
        #将文件命加入到当前文件路径后面
        #print(newDir)
        if os.path.isfile(newDir) :         #如果是文件
            if os.path.splitext(newDir)[1]==".txt":  #判断是否是txt
                if 'A' in newDir:
                    sceneList.append(newDir)
                elif 'B' in newDir:
                    threatList.append(newDir)                              
                pass
        else:
            eachFile(newDir)                #如果不是文件，递归这个文件夹的路径
    return sceneList, threatList 
 
def pathFile(filepath):
    #F：读路径文件名列表
    #
    pathDir = os.listdir(filepath)
    pathList = []
    for s in pathDir:
        newDir=os.path.join(filepath,s)
        pathList.append(newDir)
    return pathList    
    
def pre_path1():
    pathList = pathFile(fileq)
    size0 = len(pathList)
    
    for i in range(size0):
        path = np.loadtxt(pathList[i],skiprows=1)
        end = path.shape[0]
        path_result = path[:,:]*1
        np.savetxt('./pathfolder/20190326000'+ '%04d'%i +'.txt', path_result, fmt='%s', newline='\r\n')
    return

#def pre_extend():
    #sceneList, threatList = eachFile(filep)
    #if len(sceneList) == len(threatList):
        #size = len(sceneList)
    #else:
        #print('False')
      
    #for i in range(size):
        #scene = np.loadtxt(sceneList[i])    
        #threat = np.loadtxt(threatList[i])
        #s_extend = chazhi.chazhi(scene, 5)
        #t_extend = chazhi.chazhi(threat, 5)
        #np.savetxt('./environment/2018032600'+ '%04d'%i +'A.txt', s_extend, fmt='%s', newline='\r\n')
        #np.savetxt('./environment/2018032600'+ '%04d'%i +'B.txt', t_extend, fmt='%s', newline='\r\n')
    #return 


print(eachFile(filep))
pre_path1()


