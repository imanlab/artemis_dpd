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
  points_MF = np.array(points_MF )
  return points_MF

def fromMF2BF (data,marker):
  n,m = np.shape(data)
  points_BF = []
  T = transfor_matrix(marker)
  for i in range(n):
    point_MF = np.append(data[i],[1])
    point_BF = np.dot(T,point_MF)
    points_BF.append(point_BF[0:3])
  points_BF = np.array(points_BF )
  return points_BF

def path(n):

	file_path = (PALPATION_DATA +"/newdata%s.json" %(n))
	with open(file_path) as f:
		data = json.loads(f.read())

	marker = np.load(MARKER_PATH)


	if n<15:
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
	#index2 = index[a[0]+1]
	index2 = index[a[-1]+1]

	xyz = fromBF2MF(EE_pos[index1:index2,0:3],marker)
	return xyz[0::50,0], Time[index1:index2:50]


with tf.device('/cpu:0'):

	dataset = []

	#T=250
	T=300

	n_trajectory = 31
	last_num = 0

	for n_samples in range(1,n_trajectory+1):
		ysamples1, time = path(n_samples)
		time_norm = (time-time[0])/(time[-1]-time[0])
		tsamples1 = np.linspace(0,1, T)
		#yinterp = np.interp(tsamples1, time_norm, ysamples1)
		#last_num = yinterp[-1]
		yinterp = ysamples1 +last_num
		last_num = yinterp[-1]


		for j in yinterp:

			dataset.append(np.expand_dims(j,axis=0))

	dataset = np.array(dataset)
	print(np.shape(dataset))

	# split into train and test sets
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size

	train, test = np.array(dataset[0:train_size]), np.array(dataset[train_size:len(dataset)])
	print('train shape',len(train),'test shape', len(test))


	validation_set_size = 700

	train_set = train[0:train.shape[0]-validation_set_size , : ]
	validation_set = train[train.shape[0]-validation_set_size: , : ]
	test_set = test[: , :]

	#test_set[:,-1] = test_set[:,-1]-test_set[0,-1]

	timeWindow = 50
	timeprediction = 50

	# reshape into X=t and Y=t+1
	trainX, trainY = create_dataset(train_set, timeWindow,timeprediction)
	validationX, validationY = create_dataset(validation_set,timeWindow,timeprediction)
	testX, testY = create_dataset(test_set, timeWindow,timeprediction)

	print('validation',validationX.shape)
	print('trainX',trainX.shape)
	print('trainY',trainY.shape)

	monitor = EarlyStopping(monitor='loss', min_delta=1e-4, patience=5, verbose=1, mode='auto', restore_best_weights=True)

	inputState = keras.layers.Input(shape=(timeWindow,1))
	z = keras.layers.TimeDistributed(keras.layers.Dense(150, activation="relu"))(inputState)
	z = Model(inputs=inputState, outputs=z)
	TCN_Model = TCN(nb_filters=150, kernel_size=2, nb_stacks=1, dilations=[1, 2, 4, 8],return_sequences=True, activation='relu')(z.output)
	TCN_Model = keras.layers.TimeDistributed(keras.layers.Dense(1,activation="linear"))(TCN_Model)

	model4 = Model(inputs=inputState, outputs=TCN_Model)
	model4.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam())
	history =  model4.fit(trainX ,trainY,  epochs=15, validation_data=(validationX[:,:,-1], validationY), verbose=2)

	inputState1 = keras.layers.Input(shape=(timeWindow,1))
	z1 = keras.layers.TimeDistributed(keras.layers.Dense(150, activation="relu"))(inputState1)
	z1 = Model(inputs=inputState1, outputs=z1)
	LSTM_Model = keras.layers.LSTM(150, return_sequences=True, activation='relu')(z1.output)
	LSTM_Model = keras.layers.TimeDistributed(keras.layers.Dense(1,activation="linear"))(LSTM_Model)

	model1 = Model(inputs=inputState1, outputs=LSTM_Model)
	model1.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam())
	model1.fit(trainX ,trainY,  epochs=15, validation_data=(validationX[:,:,-1], validationY), verbose=2)



	inputState2 = keras.layers.Input(shape=(timeWindow,1))
	z2 = keras.layers.TimeDistributed(keras.layers.Dense(150, activation="relu"))(inputState2)
	z2 = Model(inputs=inputState2, outputs=z2)
	GRU_Model = keras.layers.GRU(150, return_sequences=True, activation='relu')(z2.output)
	GRU_Model = keras.layers.TimeDistributed(keras.layers.Dense(1,activation="linear"))(GRU_Model)

	model2 = Model(inputs=inputState2, outputs=GRU_Model)
	model2.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam())
	model2.fit(trainX ,trainY,  epochs=15, validation_data=(validationX[:,:,-1], validationY),
					verbose=2)

	inputState3 = keras.layers.Input(shape=(timeWindow,1))
	z3 = keras.layers.TimeDistributed(keras.layers.Dense(150, activation="relu"))(inputState3)
	z3 = Model(inputs=inputState3, outputs=z3)
	RNN_model = keras.layers.SimpleRNN(150, return_sequences=True, activation='relu')(z3.output)
	RNN_model = keras.layers.TimeDistributed(keras.layers.Dense(1,activation="linear"))(RNN_model)

	model3 = Model(inputs=inputState3, outputs=RNN_model)
	model3.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam())
	model3.fit(trainX ,trainY,  epochs=15, validation_data=(validationX[:,:,-1], validationY),
					verbose=2)

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss X axis')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig(PLOTS + 'loss_X.png')

	test_predict = model4.predict(testX[:,:,-1])
	test_predict1 = model1.predict(testX[:,:,-1])
	test_predict2 = model2.predict(testX[:,:,-1])
	test_predict3 = model3.predict(testX[:,:,-1])


	target = np.zeros(shape=(np.shape(test_predict)[0]))

	predict = np.zeros(shape=(np.shape(test_predict)[0]))
	predict1 = np.zeros(shape=(np.shape(test_predict)[0]))
	predict2 = np.zeros(shape=(np.shape(test_predict)[0]))
	predict3 = np.zeros(shape=(np.shape(test_predict)[0]))


	for i in range(np.shape(test_predict)[0]):
		target[i] = testY[i,0]

		predict[i] = test_predict[i,0]
		predict1[i] = test_predict1[i,0]
		predict2[i] = test_predict2[i,0]
		predict3[i] = test_predict3[i,0]

	print('target shape:', np.shape(target))
	print('predict shape:', np.shape(predict))


	distance, path = fastdtw(predict, target, dist=euclidean)
	distance1, path = fastdtw(predict1, target, dist=euclidean)
	distance2, path = fastdtw(predict2, target, dist=euclidean)
	distance3, path = fastdtw(predict3, target, dist=euclidean)


	print('TCN',distance,'LSTM', distance1,'GRU',distance2,'RNN',distance3)

	plt.figure()
	plt.plot(predict[:500], '--r', label='Prediction TCN')
	plt.plot(predict1[:500], '--b', label='Prediction LSTM')
	plt.plot(predict2[:500], '--k', label='Prediction GRU')
	plt.plot(predict3[:500], '--', label='Prediction RNN')


	plt.plot(target[:500], '--g' , label='GT' )
	plt.ylabel('X axis')
	plt.xlabel('n_samples')
	plt.title('Prediction in test phase X axis')
	plt.legend( loc='upper left')
	plt.legend()

	plt.savefig(PLOTS + 'X1_TCN_new2.png')

	plt.figure()
	plt.plot(predict[150:250], '--r', label='Prediction TCN')
	plt.plot(predict1[150:250], '--b', label='Prediction LSTM')
	plt.plot(predict2[150:250], '--k', label='Prediction GRU')
	plt.plot(predict3[150:250], '--', label='Prediction RNN')


	plt.plot(target[150:250], '--g' , label='GT' )
	plt.ylabel('X axis')
	plt.xlabel('n_samples')
	plt.title('Prediction in test phase Y axis')
	plt.legend( loc='upper left')
	plt.legend()

	plt.savefig(PLOTS + 'X1_TCN_new2zoom.png')


	error = np.mean(abs(predict-target))
	print(error)



	#model2.save('/home/kiyanoushs/Marta/ARTEMIS/MODEL/ultima_spiaggia/GRUX.h5')


	# model.summary()
