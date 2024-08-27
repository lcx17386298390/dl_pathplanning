# -*- coding: utf-8 -*-

import tkinter
import tkinter.messagebox
import resnet50
import zonghe
import os
import threading
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
def thread_it(func, *args):
    '''将函数打包进线程'''
    # 创建
    t = threading.Thread(target=func, args=args) 
    # 守护 !!!
    t.setDaemon(True) 
    # 启动
    t.start()
    # 阻塞--卡死界面！
filep = 'C:/Users/lenovo/Desktop/615/xun/s/' #"训练集场景信息文件位置"
fileq = 'C:/Users/lenovo/Desktop/615/test/s/' #"测试集场景信息文件位置"
def count1(filepath):
    pathDir = os.listdir(filepath)      #获取当前路径下的文件名，返回List
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
    return len(sceneList)
def show6():
    text1.delete(0.0,"end")
    text1.insert(1.0,"正在进行航路规划......")
    zonghe.hangji()
    text1.insert(2.0,"航路规划已完成")
def show5():
    text1.delete(0.0,"end")
    text1.insert(1.0,"深度神经网络模型正在训练中......")
    text1.insert(2.0,resnet50.kaishi())
    text1.insert(3.0,"训练模型已完成")
def show1():
    text1.delete(0.0,"end")
    text1.insert(1.0,"训练集文件数目为")
    text1.insert(2.0,count1(filep))
def show2():
    text1.delete(0.0,'end')
    text1.insert(1.0,"测试集文件数目为")
    text1.insert(2.0,count1(fileq))
def show3():
    text1.delete(0.0,'end')
    path='./B/2019022800000000.png' #"准确率文件存储位置"
    photo=Image.open(path)
    photo.show()
def show4():
    path='./A/2019022800000000.png' #"航路规划生成文件位置"
    photo=Image.open(path)
    photo.show()
top=tkinter.Tk()
top.title('深度学习控制模块')
top.geometry("580x300")
p1 = tkinter.PhotoImage(file ='1.png')
p2 = tkinter.PhotoImage(file ='2.png')
p3 = tkinter.PhotoImage(file ='3.png')
p4 = tkinter.PhotoImage(file ='45.png')
p5 = tkinter.PhotoImage(file ='6.png')
p6 = tkinter.PhotoImage(file ='7.png') 
frame = tkinter.Frame(top,width=580,height=0.5)
bt1 = tkinter.Button(frame,image=p1,text="训练",font=("Calibri",9),compound = "left",command=lambda:thread_it(show5))
bt1.pack(side="left") 
bt2 = tkinter.Button(frame,text="航迹规划",image=p2,compound = "left",font=("Calibri",9),command=lambda:thread_it(show6))
bt2.pack(side="left",) 
bt3 = tkinter.Button(frame,text="停止",image=p3,compound = "left",font=("Calibri",9),command=lambda:thread_it(os._exit(0)))
bt3.pack(side="left",) 
bt4 = tkinter.Button(frame,text="训练文件",image=p4,compound = "left",font=("Calibri",9),command=lambda:thread_it(show1))
bt4.pack(side="left",) 
bt5 = tkinter.Button(frame,text="测试文件",image=p4,compound = "left",font=("Calibri",9),command=lambda:thread_it(show2))
bt5.pack(side="left",) 
bt6 = tkinter.Button(frame,text="准确率",image=p5,compound = "left",font=("Calibri",9),command=lambda:thread_it(show3))
bt6.pack(side="left",) 
bt7 = tkinter.Button(frame,text="航迹规划结果",image=p6,compound = "left",font=("Calibri",9),command=lambda:thread_it(show4))
bt7.pack(side="left",) 
frame.pack(side="top",fill="x") 
menu=tkinter.Menu(top)
top.config(menu=menu)
menu1=tkinter.Menu(menu,tearoff=0)
menu1.add_command(label="训练模型",command=lambda:thread_it(show5))
menu1.add_command(label="停止进程",command=lambda:thread_it(os._exit(0)))
menu.add_cascade(label="菜单",menu=menu1)
menu3=tkinter.Menu(menu,tearoff=0)
menu3.add_command(label="航迹规划",command=lambda:thread_it(show6))
menu.add_cascade(label="功能",menu=menu3)
menu2=tkinter.Menu(menu,tearoff=0)
menu2.add_command(label="训练集文件数目",command=lambda:thread_it(show1))
menu2.add_command(label="测试集文件数目",command=lambda:thread_it(show2))
menu2.add_command(label="训练准确率反馈",command=lambda:thread_it(show3))
menu2.add_command(label="航迹规划结果查询",command=lambda:thread_it(show4))
menu.add_cascade(label="查询",menu=menu2)
text2=tkinter.Text(top,width=580,height=1,background="Black",font=("Calibri",15),fg='GhostWhite')
text2.insert(1.0,"程序执行结果:")
text2.pack()
text1=tkinter.Text(top,width=580,height=12,background="Black",font=("Calibri",15),fg='GhostWhite')
text1.pack()
top.mainloop()