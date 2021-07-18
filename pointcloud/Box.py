#!/usr/bin/env python

import numpy as np 
import pandas as pd
import tf
import rospy
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import struct

import open3d as o3d

 

def transfor_matrix(ref):
	trans = ref[0:3]
	rot = ref[3::]
	
	r = R.from_quat(rot)
	Rmatrix = r.as_dcm()
	Rmatrix = np.vstack([Rmatrix,[0,0,0]])
	trans = np.append(trans,[1])
	
	T_matrix = np.transpose(np.vstack([np.transpose(Rmatrix),trans]))
	
	return T_matrix

def point_phantom (data, marker):
	n, m = np.shape(data)
	points_BF = []

	
	
	T = inv(transfor_matrix(marker))
	# B = borders_breast()
	B = borders_phantom()


	for i in range (n):
		point_BF = np.append(data[i],[1])
		point_MF = np.dot(T,point_BF)

		if point_MF[0]<=B[1,0] and point_MF[0]>=B[0,0] and point_MF[1]<=B[1,1] and point_MF[1]>=B[0,1]:
			points_BF.append(data[i])
	# z_max = np.max(data[:,2])
	# final = []
	# for i in points_BF:
	# 	if i[2]>z_max-0.005:
	# 		final.append(i)



	return points_BF

def borders_phantom():
	b = np.array([[-0.015,0.025],[0.125,0.165]])
	return b

def borders_breast():
	b = np.array([[-0.08,-0.27],[0.098,-0.02]])
	return b





if __name__ == '__main__':

	rospy.init_node('segmentation', anonymous=False)

	n=12


	data1 = pd.read_csv(('/home/kiyanoush/catkin_ws/src/camera_calibration/scripts/Data/PointCloudData/%s/pointcloud%s_1.csv' %(n,n)), header=None)
	# data2 = pd.read_csv(('/home/kiyanoush/catkin_ws/src/camera_calibration/scripts/Data/PointCloudData/%s/pointcloud%s_2.csv' %(n,n)), header=None)
	# data3 = pd.read_csv(('/home/kiyanoush/catkin_ws/src/camera_calibration/scripts/Data/PointCloudData/%s/pointcloud%s_3.csv' %(n,n)), header=None)
	# data4 = pd.read_csv(('/home/kiyanoush/catkin_ws/src/camera_calibration/scripts/Data/PointCloudData/%s/pointcloud%s_4.csv' %(n,n)), header=None)
	# data5 = pd.read_csv(('/home/kiyanoush/catkin_ws/src/camera_calibration/scripts/Data/PointCloudData/%s/pointcloud%s_5.csv' %(n,n)), header=None)

	
	# data = np.concatenate((data1,data2,data3,data4,data5))
	# data = np.concatenate((data1))
	data = np.array(data1)
	print(np.shape(data))


	marker = pd.read_csv(('/home/kiyanoush/catkin_ws/src/camera_calibration/scripts/Data/MarkerPose/%s/MarkerPose%s_1.csv' %(n,n)), header=None).to_numpy()
	
	points_phantom = point_phantom(data,marker.reshape((-1)))  
	print(np.shape(points_phantom))
	df = pd.DataFrame(points_phantom)
	df.to_csv(('/home/kiyanoush/catkin_ws/src/camera_calibration/scripts/Data/PointCloudData/phantom_PC/phantom%s.csv' %(n)), header= None, index= None)

	
