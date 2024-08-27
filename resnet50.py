#import pandas as pd
import bigbatch
import tensorflow 
import numpy as nppip 
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras._impl.keras.utils import layer_utils
from tensorflow.python.keras._impl.keras.utils.data_utils import get_file
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import preprocess_input
import pydotplus
#from IPython.display import SVG
from tensorflow.python.keras._impl.keras.utils.vis_utils import model_to_dot
from tensorflow.python.keras.utils import plot_model
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.python.keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from tensorflow.python.keras.layers import Dropout
# from tensorflow.keras.utils import np_utils 
import bbbbb
import matplotlib.pyplot as plt
import h5py

from tensorflow.python.keras import backend as K
import tensorflow as tf
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def identity_block(X,f,filters,stage,block):
    #defining name basis
    conv_name_base = 'res'+str(stage)+block+'_branch'
    bn_name_base = 'bn'+str(stage)+block+'_branch'
    
    #Retrive Filters
    F1,F2,F3=filters
    
    # Save the input value.
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    #Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    #Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    
    #Final step:
    
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)

    return X

#tf.reset_default_graph()
#
#with tf.Session() as test:
#    np.random.seed(1)
#    A_prev = tf.placeholder("float",[3,4,4,6])
#    X = np.random.randn(3,4,4,6)
#    A = identity_block(A_prev, f = 2, filters = [2,4,6],stage = 1, block = 'a')
#    test.run(tf.global_variables_initializer())
#    out = test.run([A],feed_dict={A_prev: X, K.learning_phase(): 0})
#    print("out = " + str(out[0][1][1][0]))
    
def convolutional_block(X, f, filters, stage, block, s=2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    ### END CODE HERE ###
    return X    
    
#tf.reset_default_graph()
#
#with tf.Session() as test:
#    np.random.seed(1)
#    A_prev = tf.placeholder("float", [3, 4, 4, 6])
#    X = np.random.randn(3, 4, 4, 6)
#    A = convolutional_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a')
#    test.run(tf.global_variables_initializer())
#    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
#    print("out = " + str(out[0][1][1][0]))    
    

def ResNet50(input_shape = (32, 32, 2), classes = 8):
    X_input = Input(input_shape)
    
    X = ZeroPadding2D((3,3))(X_input)

    # Stage 1
    #修改7*7；（2,2） （2,2）
    X = Conv2D(64, (7, 7), strides=(1, 1), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(1, 1))(X)


    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    #Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    #Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    #X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    #X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    #stage 6
    #X = X = convolutional_block(X, f=3, filters=[1024, 1024, 4096], stage=6, block='a', s=2)
    #X = identity_block(X, 3, [1024, 1024, 4096], stage=6, block='b')
    #X = identity_block(X, 3, [1024, 1024, 4096], stage=6, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    #X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)j
    X = Dense(classes, activation='softmax',name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')
    

    return model


def kaishi():
#     model = ResNet50(input_shape = (32, 32, 2), classes = 8)
# #model = load_model('resNet50_a-starall.h5')
# #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# #X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# #
# ## Normalize image vectors
# #X_train = X_train_orig / 255.
# #X_test = X_test_orig / 255.
# #
# ## Convert training and test labels to one hot matrices
# #Y_train = convert_to_one_hot(Y_train_orig, 6).T
# #Y_test = convert_to_one_hot(Y_test_orig, 6).T
#     X_train, Y_train = bigbatch.pre()
#     print("number of training examples = " + str(X_train.shape[0]))    
#     print("X_train shape: " + str(X_train.shape))
#     print("Y_train shape: " + str(Y_train.shape))
#     X_val, Y_val = bbbbb.pre()
#     for i in range(16):
#         p=[]
#         q=[]
#         model.fit(X_train, Y_train, epochs = 1, batch_size = 64)
#         history=model.evaluate(X_val, Y_val, batch_size=64)
#         p.append(history[1])
#         q.append(i)
#         plt.plot()
#         plt.plot(q,p,"r-", marker='o', linewidth = 1)
#         plt.title('model accuracy')
#         plt.ylabel('accuracy')
#         plt.xlabel('epoch')
#         plt.show()
#         plt.savefig('./B/2019022800000000'+'.png') #准确率存储位置
#         plt.close('all')
#         if i%3 == 0:
#             model.save('resNet.h5') #模型存储名称
#         print(i)
#         return(i)

    # 2024-6-7修改
    # graph = tf.Graph()
    # with graph.as_default():
    #     model = ResNet50(input_shape = (32, 32, 2), classes = 8)
    #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # 用的是adam优化器，使用的keras版本要高
    #     X_train, Y_train = bigbatch.pre()
    #     X_val, Y_val = bbbbb.pre()
    #     for i in range(16):
    #         p=[]
    #         q=[]
    #         model.fit(X_train, Y_train, epochs = 10, batch_size = 64)
    #         history=model.evaluate(X_val, Y_val, batch_size=64)
    #         p.append(history[1])    # p存储的是准确率
    #         q.append(i)     # q存储的是迭代次数，即epoch
    #         plt.plot()
    #         plt.plot(q,p,"r-", marker='o', linewidth = 1)   
    #         plt.title('model accuracy')
    #         plt.ylabel('accuracy')
    #         plt.xlabel('epoch')
    #         plt.savefig('./B/2019022800000000'+'.png') #准确率存储位置
    #         plt.show()
    #         plt.close('all')
    #         if i%3 == 0:
    #             model.save('resNet_240618.h5') #模型存储名称
    #         print(i)
    #         return(i)

    # 2024-6-19修改
    graph = tf.Graph()
    with graph.as_default():
        model = ResNet50(input_shape=(32, 32, 2), classes=8)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        X_train, Y_train = bigbatch.pre()
        X_val, Y_val = bbbbb.pre()
        p = []  # 移动到循环外部
        q = []  # 移动到循环外部
        for i in range(16):
            model.fit(X_train, Y_train, epochs=10, batch_size=64)
            history = model.evaluate(X_val, Y_val, batch_size=64)
            p.append(history[1])  # 存储准确率
            q.append(i)  # 存储迭代次数
            if i % 3 == 0:
                model.save('resNet_240618.h5')  # 模型存储名称
            print(i)
        # 循环结束后绘制图表
        plt.plot(q, p, "r-", marker='o', linewidth=1)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('./B/2019022800000000'+'.png')  # 准确率存储位置
        plt.show()
        plt.close('all')
        return i  # 移动到循环外部
