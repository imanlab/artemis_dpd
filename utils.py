import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import json, os, sys
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

current_dir = os.getcwd()
sys.path.insert(1, current_dir)
from config import *

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2regmarker
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        return {"num_features": self.num_features,"l2reg": self.l2reg}

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, look_forword=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1-look_forword):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back:i + look_back+look_forword, -1])
    return np.array(dataX), np.array(dataY)

def transfor_matrix(ref):
    trans = ref[0:3]
    rot = ref[3::]

    r = R.from_quat(rot)
    Rmatrix = r.as_matrix()
    Rmatrix = np.vstack([Rmatrix,[0,0,0]])
    trans = np.append(trans,[1])

    T_matrix = np.transpose(np.vstack([np.transpose(Rmatrix),trans]))
    return T_matrix

def fromBF2MF(data,marker):
    n,m = np.shape(data)
    points_MF = []
    T = np.linalg.inv(transfor_matrix(marker))
    for i in range(n):
        point_BF = np.append(data[i],[1])
        point_MF = np.dot(T,point_BF)
        points_MF.append(point_MF[0:3])
    points_MF = np.array(points_MF)
    return points_MF

def fromMF2BF (data,marker):
    n,m = np.shape(data)
    points_BF = []
    T = transfor_matrix(marker)
    for i in range(n):
        point_MF = np.append(data[i],[1])
        point_BF = np.dot(T,point_MF)
        points_BF.append(point_BF[0:3])
    points_BF = np.array(points_BF)
    return points_BF

def path(n,coordinate):
    file_path = (PALPATION_DATA +"/newdata%s.json" %(n))
    with open(file_path) as f:
        data = json.loads(f.read())

    marker = np.load(MARKER_PATH)

    if n < 15:
        marker = np.array(marker)[0]
    else:
        marker = np.array(marker)[4]

    Time = np.array(data['time'])[200:-200]
    Normal = np.array(data['Normal'])[200:-200,:]
    Transf_matrix = np.array(data['transf_matrix'])[200:-200]
    EE_pos = Transf_matrix[:,0:3,3]

    Normal_mean = np.mean(Normal,axis=1)
    Normal_nor = Normal_mean/np.max(Normal_mean)-np.min(Normal_mean[2000::]/np.max(Normal_mean))

    count = 0
    index = []
    for i in Normal_nor:
        if abs(i)>0.007:
            index.append(count)
        count+=1

    index = np.array(index)
    diff = index[1::]-index[:-1]
    index1 = index[0]
    a = np.array(np.where(diff != 1)[0])
    index2 = index[a[-1]+1]

    xyz = fromBF2MF(EE_pos[index1:index2,0:3],marker)
    return xyz[0::50,coordinate], Time[index1:index2:50] # 0 here is X traj
