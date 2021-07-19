import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tcn import TCN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import json, os, sys

current_dir = os.getcwd()
sys.path.insert(1, current_dir)
from config import *

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, look_forword=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1-look_forword):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back:i + look_back+look_forword])
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

def path(n, coordinate):
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

	return xyz[0::50,coordinate], Time[index1:index2:50]

with tf.device('/gpu:1'):
	dataset = []

	T=300
	timeWindow = 50
	timeprediction = 50
	n_trajectory = 14
	last_num = 0

	mediax = np.zeros(timeWindow)
	mediay = np.zeros(timeWindow)
	mediaz = np.zeros(timeWindow)

	for n_samples in range(1,n_trajectory+1):
		xsamples1, time = path(n_samples,0)
		ysamples1, time = path(n_samples,1)
		zsamples1, time = path(n_samples,2)
		time_norm = (time-time[0])/(time[-1]-time[0])
		tsamples1 = np.linspace(0,1, T)

		xinterp = np.interp(tsamples1, time_norm, xsamples1)
		yinterp = np.interp(tsamples1, time_norm, ysamples1)
		zinterp = np.interp(tsamples1, time_norm, zsamples1)


		last_num = yinterp[-1]
		#mediax = mediax + xinterp
		#mediay = mediay + yinterp
		#mediaz = mediaz + zinterp

		mediax = xsamples1[:timeWindow] + mediax
		mediay = xsamples1[:timeWindow] + mediay
		mediaz = xsamples1[:timeWindow] + mediaz

		for j in yinterp:
			dataset.append(j)

	dataset = np.array(dataset)
	print(np.shape(dataset))

	# split into train and test sets
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size

	train, test = np.array(dataset[0:train_size]), np.array(dataset[train_size:len(dataset)])
	print('train shape',len(train),'test shape', len(test))

	validation_set_size = 1
	train_set = train[0:train.shape[0]-validation_set_size ]
	validation_set = train[train.shape[0]-validation_set_size:]
	test_set = test[:]

	# reshape into X=t and Y=t+1
	trainX, trainY = create_dataset(dataset, timeWindow,timeprediction)
	_ , _ = create_dataset(validation_set,timeWindow,timeprediction)
	_ , _ = create_dataset(test_set, timeWindow,timeprediction)

	print('trainX',trainX.shape)
	print('trainY',trainY.shape)

	modelz = keras.models.load_model(current_dir + "/models/Z_TCN.h5", custom_objects={'TCN': TCN})
	modely = keras.models.load_model(current_dir + "/models/Y_TCN.h5", custom_objects={'TCN': TCN})
	modelx = keras.models.load_model(current_dir + "/models/X_TCN.h5", custom_objects={'TCN': TCN})
	modelx.summary()
	modely.summary()
	modelz.summary()

	#initial_trajectory = initial15(start_trajectory)
			#it = T - timeWindow
	it = 200

	targetx = np.zeros(shape=(it))
	targety = np.zeros(shape=(it))
	targetz = np.zeros(shape=(it))

	x = True
	iteration = 0

	seqx = xsamples1[:timeWindow]
	seqy = ysamples1[:timeWindow]
	seqz = zsamples1[:timeWindow]

	while x:

		predictionx = modelx.predict(np.expand_dims(np.transpose(np.expand_dims(seqx,axis=0)),axis=0))
		predictiony = modely.predict(np.expand_dims(np.transpose(np.expand_dims(seqy,axis=0)),axis=0))
		predictionz = modelz.predict(np.expand_dims(np.transpose(np.expand_dims(seqz,axis=0)),axis=0))
		#print(np.shape(predictionx),'shape prediction X')

		targetx[iteration] = predictionx[0,0]
		targety[iteration] = predictiony[0,0]
		targetz[iteration] = predictionz[0,0]

		next_stepx = predictionx[0,0]
		next_stepy = predictiony[0,0]
		next_stepz = predictionz[0,0]

		iteration+=1

		seqx[0:-1] = seqx[1::]
		seqx[-1] = next_stepx

		seqy[0:-1] = seqy[1::]
		seqy[-1] = next_stepy

		seqz[0:-1] = seqz[1::]
		seqz[-1] = next_stepz

		if iteration >= it:
			x=False
			print('finish')

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.plot(targetx, targety, targetz)

	plt.savefig(PLOTS + '/trackingprova_GRU3D.png')

	plt.figure()

	plt.plot(targetx, targety, 'b')
	plt.savefig(PLOTS + '/trackingprova_GRU.png')

	plt.figure()
	plt.plot(targetx, 'ob',label='Prediction X')
	plt.ylabel('X axis')
	plt.xlabel('n_samples')
	plt.title('Prediction real time X axis')
	plt.legend( loc='upper left')
	plt.legend()
	plt.savefig(PLOTS + '/trackingX_GRU.png')


	plt.figure()
	plt.plot(targety, 'ob',label='Prediction Y')
	plt.ylabel('Y axis')
	plt.xlabel('n_samples')
	plt.title('Prediction real time Y axis')
	plt.legend( loc='upper left')
	plt.legend()
	plt.savefig(PLOTS + '/trackingY_GRU.png')


	plt.figure()
	plt.plot(targetz, 'ob',label='Prediction Z')
	plt.ylabel('Z axis')
	plt.xlabel('n_samples')
	plt.title('Prediction real time Z axis')
	plt.legend(loc='upper left')
	plt.legend()
	plt.savefig(PLOTS + '/trackingZ_GRU.png')
