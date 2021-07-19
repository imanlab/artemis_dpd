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
from utils import *

with tf.device('/cpu:0'):
    dataset = []

    T=300
    n_trajectory = 31
    last_num = 0

    for n_samples in range(1,n_trajectory+1):
        ysamples1, time = path(n_samples,1)
        time_norm = (time-time[0])/(time[-1]-time[0])
        tsamples1 = np.linspace(0,1, T)
        #yinterp = np.interp(tsamples1, time_norm, ysamples1)
        yinterp = ysamples1
        last_num = yinterp[-1]

        # print(len(yinterp), 'lunghezza trajectory %s' %(n_samples))
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

    """
    TCN
    """
    inputState = keras.layers.Input(shape=(timeWindow,1))
    z = keras.layers.TimeDistributed(keras.layers.Dense(150, activation="relu"))(inputState)
    z = Model(inputs=inputState, outputs=z)
    TCN_Model = TCN(nb_filters=150, kernel_size=2, nb_stacks=1, dilations=[1, 2, 4, 8],return_sequences=True, activation='relu')(z.output)
    TCN_Model = keras.layers.TimeDistributed(keras.layers.Dense(1,activation="linear"))(TCN_Model)

    model4 = Model(inputs=inputState, outputs=TCN_Model)
    model4.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam())
    history4 =  model4.fit(trainX ,trainY,  epochs=15, validation_data=(validationX[:,:,-1], validationY), verbose=2)

    """
    LSTM
    """
    inputState1 = keras.layers.Input(shape=(timeWindow,1))
    z1 = keras.layers.TimeDistributed(keras.layers.Dense(150, activation="relu"))(inputState1)
    z1 = Model(inputs=inputState1, outputs=z1)
    LSTM_Model = keras.layers.LSTM(150, return_sequences=True, activation='relu')(z1.output)
    LSTM_Model = keras.layers.TimeDistributed(keras.layers.Dense(1,activation="linear"))(LSTM_Model)

    model1 = Model(inputs=inputState1, outputs=LSTM_Model)
    model1.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam())
    history1 = model1.fit(trainX ,trainY,  epochs=15, validation_data=(validationX[:,:,-1], validationY), verbose=2)

    """
    GRU
    """
    inputState2 = keras.layers.Input(shape=(timeWindow,1))
    z2 = keras.layers.TimeDistributed(keras.layers.Dense(150, activation="relu"))(inputState2)
    z2 = Model(inputs=inputState2, outputs=z2)
    GRU_Model = keras.layers.GRU(150, return_sequences=True, activation='relu')(z2.output)
    GRU_Model = keras.layers.TimeDistributed(keras.layers.Dense(1,activation="linear"))(GRU_Model)

    model2 = Model(inputs=inputState2, outputs=GRU_Model)
    model2.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam())
    history2 = model2.fit(trainX ,trainY,  epochs=15, validation_data=(validationX[:,:,-1], validationY), verbose=2)

    """
    RNN
    """
    inputState3 = keras.layers.Input(shape=(timeWindow,1))
    z3 = keras.layers.TimeDistributed(keras.layers.Dense(150, activation="relu"))(inputState3)
    z3 = Model(inputs=inputState3, outputs=z3)
    RNN_model = keras.layers.SimpleRNN(150, return_sequences=True, activation='relu')(z3.output)
    RNN_model = keras.layers.TimeDistributed(keras.layers.Dense(1,activation="linear"))(RNN_model)

    model3 = Model(inputs=inputState3, outputs=RNN_model)
    model3.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam())
    history3 = model3.fit(trainX ,trainY,  epochs=15, validation_data=(validationX[:,:,-1], validationY), verbose=2)

    """
    PLOTS
    """
    
    plt.plot(history4.history['loss'])
    plt.plot(history4.history['val_loss'])
    plt.title('model loss Y axis')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(PLOTS + '/Y_loss.png')

    test_predict = model4.predict(testX[:,:,-1]) # TCN
    test_predict1 = model1.predict(testX[:,:,-1]) # LSTM
    test_predict2 = model2.predict(testX[:,:,-1]) # GRU
    test_predict3 = model3.predict(testX[:,:,-1]) # RNN

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
    plt.ylabel('Y axis')
    plt.xlabel('n_samples')
    plt.title('Prediction in test phase Y axis')
    plt.legend(loc='upper left')
    plt.legend()

    plt.savefig(PLOTS + '/Y_TCN_new2.png')

    plt.figure()
    plt.plot(predict[150:250], '--r', label='Prediction TCN')
    plt.plot(predict1[150:250], '--b', label='Prediction LSTM')
    plt.plot(predict2[150:250], '--k', label='Prediction GRU')
    plt.plot(predict3[150:250], '--', label='Prediction RNN')

    plt.plot(target[150:250], '--g' , label='GT' )
    plt.ylabel('Y axis')
    plt.xlabel('n_samples')
    plt.title('Prediction in test phase Y axis')
    plt.legend( loc='upper left')
    plt.legend()

    plt.savefig(PLOTS + '/Y_TCN_new2zoom.png')

    error = np.mean(abs(predict-target))
    print(error)

    model1.save(TCN_MODEL + '/Y_LSTM.h5')
    model2.save(TCN_MODEL + '/Y_GRU.h5')
    model3.save(TCN_MODEL + '/Y_RNN.h5')
    model4.save(TCN_MODEL + '/Y_TCN.h5')
