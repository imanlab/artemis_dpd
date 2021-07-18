#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_msgs.msg import Int32
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import MultiArrayLayout
from camera_calibration.msg import point
import tf
import numpy as np
from scipy.spatial.transform import Rotation as R
import ast
import time
import pandas as pd

class ListenAndPublish(object):

    def __init__(self):

        self.listener = tf.TransformListener()
        rospy.Subscriber("points", point, self.callback)
        # rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.callback)
        

    def get_transform(self, from_tf, to_tf):
        self.listener.waitForTransform(from_tf, to_tf, rospy.Time(), rospy.Duration(4.0))
        return self.listener.lookupTransform(from_tf, to_tf, rospy.Time())      

    def callback(self, data):
        # self.points = np.fromstring(data.data, dtype = float)
        # self.points = np.asarray(data.xyz)
        points_BF = []
        (trans, rot) = self.get_transform('panda_link0', 'camera_color_optical_frame')
        r = R.from_quat(rot)
        Rmatrix = r.as_dcm()
        Rmatrix = np.vstack([Rmatrix,[0,0,0]])
        trans.append(1)
        T_matrix = np.transpose(np.vstack([np.transpose(Rmatrix),trans]))
        
        pt_x = data.data[0:len(data.data):3]
        pt_y = data.data[1:len(data.data):3]
        pt_z = data.data[2:len(data.data):3]


        for i in range(len(pt_z)):
            point_camera = [pt_x[i],pt_y[i],pt_z[i],1]
            point_BF = np.dot(T_matrix,point_camera) 
            # points_BF.append(point_BF[0])
            # points_BF.append(point_BF[1])
            # points_BF.append(point_BF[2])
            points_BF.append(point_BF[0:-1])
        self.points_BF = np.array(points_BF)
        
    def save(self):
        #save points2
        df = pd.DataFrame(self.points_BF)
        n=12
        m=1
        df.to_csv(('/home/kiyanoush/catkin_ws/src/camera_calibration/scripts/Data/PointCloudData/%s/pointcloud%s_%s.csv' %(n,n,m)), header= None, index= None)

        #save aruco position
        (trans_m, rot_m) = self.get_transform('panda_link0', 'marker_id0')
        x = [trans_m[0],trans_m[1],trans_m[2],rot_m[0],rot_m[1],rot_m[2],rot_m[3]]

        dff = pd.DataFrame(x)
        dff.to_csv(("/home/kiyanoush/catkin_ws/src/camera_calibration/scripts/Data/MarkerPose/%s/MarkerPose%s_%s.csv" %(n,n,m)), header= None, index= None)

        

    
        
        

if __name__ == '__main__':
    
    rospy.init_node('PointCloud_baseframe', anonymous=True)
    l = ListenAndPublish()
    time.sleep(3)

    if not rospy.is_shutdown():
        l.save()
    

    rospy.spin()   


    

