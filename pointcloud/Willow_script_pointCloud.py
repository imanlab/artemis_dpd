#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import rospy
import time
import numpy as np
import pyrealsense2 as rs
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, RegionOfInterest, PointCloud2, PointField
from std_msgs.msg import MultiArrayLayout
from open3d import Image as rsImage, create_point_cloud_from_depth_image, PinholeCameraIntrinsic, PinholeCameraIntrinsicParameters
from camera_calibration.msg import point


#from std_msgs.msg import Int16MultiArray


class RealsenseCamera(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.frame_rate = 30
        # self.height = 720
        # self.width = 1280
        self.height = 480
        self.width = 640
        self.img_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('/camera/aligned_depth_to_color/image_raw', Image, queue_size=1)
        self.rgb_camera_info_pub = rospy.Publisher('/camera/color/camera_info', CameraInfo, queue_size=1)
        self.depth_camera_info_pub = rospy.Publisher('/camera/aligned_depth_to_color/camera_info', CameraInfo, queue_size=1)
        self.pointcloud_pub = rospy.Publisher('/camera/depth/color/points', PointCloud2, queue_size=1)
        self.points_CF_pub = rospy.Publisher('points', point, queue_size=1)
        # self.DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07"]
        self.DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A", "0B5C"]
        self.rate = rospy.Rate(self.frame_rate)
        self.pipeline = rs.pipeline()
        # self.profile = self.pipeline.start(self.load_config())
        self.pipeline.start(self.load_config())
        self.profile = self.pipeline.get_active_profile()
        self.align = rs.align(rs.stream.color)
        self.enable_advanced_mode()

        

        super(RealsenseCamera, self).__init__()

    def load_config(self):
        # Create a config and configure the pipeline to stream
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.frame_rate)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.frame_rate)
        return config

    def get_camera_info_msg(self, msg_header, k, p):
        cam_info = CameraInfo()
        cam_info.header = msg_header
        cam_info.header.frame_id = "/camera_color_optical_frame"
        # cam_info.height = 720
        # cam_info.width = 1280
        cam_info.height = 480
        cam_info.width = 640
        cam_info.distortion_model = "plumb_bob"
        cam_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        cam_info.K = k
        cam_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        cam_info.P = p
        cam_info.binning_x = 0
        cam_info.binning_y = 0
        cam_info.roi = RegionOfInterest()
        cam_info.roi.x_offset = 0
        cam_info.roi.y_offset = 0
        cam_info.roi.height = 0
        cam_info.roi.width = 0
        cam_info.roi.do_rectify = False
        return cam_info

    def get_rgb_info(self, msg_header):
        # k = [617.4435424804688, 0.0, 317.70989990234375, 0.0, 617.878662109375, 238.01991271972656, 0.0, 0.0, 1.0]
        # p = [617.4435424804688, 0.0, 317.70989990234375, 0.0, 0.0, 617.878662109375, 238.01991271972656, 0.0, 0.0, 0.0, 1.0, 0.0]
        k = [609.6610107421875, 0.0, 320.6727600097656, 0.0, 610.0888061523438, 246.66778564453125, 0.0, 0.0, 1.0]
        p = [609.6610107421875, 0.0, 320.6727600097656, 0.0, 0.0, 610.0888061523438, 246.66778564453125, 0.0, 0.0, 0.0, 1.0, 0.0]
        return self.get_camera_info_msg(msg_header, k, p)

    def get_depth_info(self, msg_header):
        # k = [384.4228820800781, 0.0, 320.81378173828125, 0.0, 384.4228820800781, 240.3497314453125, 0.0, 0.0, 1.0]
        # p = [384.4228820800781, 0.0, 320.81378173828125, 0.0, 0.0, 384.4228820800781, 240.3497314453125, 0.0, 0.0, 0.0, 1.0, 0.0]
        k = [385.5482177734375, 0.0, 315.8621520996094, 0.0, 385.5482177734375, 241.27386474609375, 0.0, 0.0, 1.0]
        p = [385.5482177734375, 0.0, 315.8621520996094, 0.0, 0.0, 385.5482177734375, 241.27386474609375, 0.0, 0.0, 0.0, 1.0, 0.0]

        return self.get_camera_info_msg(msg_header, k, p)

    def find_device_that_supports_advanced_mode(self):
        ctx = rs.context()
        # ds5_dev = rs.device()
        devices = ctx.query_devices()
        for dev in devices:
            if dev.sensors[0].is_depth_sensor():
                dev.sensors[0].set_option(rs.option.enable_auto_exposure, False)
                dev.sensors[0].set_option(rs.option.exposure, 2000)
                dev.sensors[0].set_option(rs.option.depth_units, 0.001)      #default unit in mm
            print(dev.supports(rs.camera_info.product_id))
            print(str(dev.get_info(rs.camera_info.product_id)))

            if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in self.DS5_product_ids:
                if dev.supports(rs.camera_info.name):
                    print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
                return dev
        raise Exception("No device that supports advanced mode was found")

    def enable_advanced_mode(self):
        dev = self.find_device_that_supports_advanced_mode()
        self.advnc_mode = rs.rs400_advanced_mode(dev)
        print("Advanced mode is", "enabled" if self.advnc_mode.is_enabled() else "disabled")

        # Loop until we successfully enable advanced mode
        while not self.advnc_mode.is_enabled():
            print("Trying to enable advanced mode...")
            self.advnc_mode.toggle_advanced_mode(True)
            # At this point the device will disconnect and re-connect.
            print("Sleeping for 5 seconds...")
            time.sleep(5)
            # The 'dev' object will become invalid and we need to initialize it again
            dev = self.find_device_that_supports_advanced_mode()
            self.advnc_mode = rs.rs400_advanced_mode(dev)

        current_std_depth_control_group = self.advnc_mode.get_depth_table()
        print (current_std_depth_control_group)
        # current_std_depth_control_group.depthUnits = 100
        # current_std_depth_control_group.disparityShift = 100

        # current_std_depth_control_group.depthClampMax = 0.5
        # current_std_depth_control_group = 0.0001


    def get_transform(self, from_tf, to_tf):
        self.listener.waitForTransform(from_tf, to_tf, rospy.Time(), rospy.Duration(4.0))
        return self.listener.lookupTransform(from_tf, to_tf, rospy.Time())      

    def generate_pointcloud(self, depth_frame, intrin):

        # # =================Peter's point cloud=====================================
        depth_frame = np.array(depth_frame.get_data())
        depth_frame[depth_frame == 0] = 10000
        intrinsics = PinholeCameraIntrinsic(PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        intrinsics.set_intrinsics(width=intrin.width, height=intrin.height, fx=intrin.fx, fy=intrin.fy, cx=intrin.ppx, cy=intrin.ppy)
        depth_frame = rsImage(depth_frame)
        # print '-----------------intrinsics', intrinsics.intrinsic_matrix
        pcd2 = create_point_cloud_from_depth_image(depth_frame, intrinsics)
        points = (np.asarray(pcd2.points, np.float32))  # / 10       # scaling division here if depth unit changed from 1000 to 100
        point_1D = np.reshape(points,-1)

        point_mes = point()
        point_mes.data = point_1D

        msg = PointCloud2()
        msg.header.frame_id = "/camera_color_optical_frame"      # depth frame has been aligned to camera_color_frame so this is our ref
        msg.height = 480
        msg.width = 640

        msg.fields = [PointField(p, 4 * s, PointField.FLOAT32, 1) for s, p in enumerate(['x', 'y', 'z'])]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False




        # Add colour to the pointcloud
        # msg.fields.append(PointField('rbg', 12, PointField.FLOAT32, 1))

        msg.data = points.tostring()

        return msg, point_mes

    def start(self):

        # Pointcloud persistency in case of dropped frames
        #pc = rs.pointcloud()
        #points = rs.points()


        try:
            while not rospy.is_shutdown():
                # Get frameset of color and depth
                frames = self.pipeline.wait_for_frames()

                # Align the depth frame to color frame
                aligned_frames = self.align.process(frames)

                # Get aligned frames
                depth_frame = aligned_frames.get_depth_frame()
                #depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                colour_frame = aligned_frames.get_color_frame()
                color_intrin = colour_frame.profile.as_video_stream_profile().intrinsics
                #print color_intrin

                # Getting the depth sensor's depth scale (see rs-align example for explanation)
                depth_sensor = self.profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                #print depth_scale
                #
                #depth_point = rs.rs2_deproject_pixel_to_point(self.color_intrin, [300,300], self.depth_scale)

                # Tell pointcloud object to map to this color frame
                #pc.map_to(colour_frame)

                # Generate the pointcloud and texture mappings
                #points = pc.calculate(depth_frame)

                # Validate that both frames are valid
                if not depth_frame or not colour_frame:
                    print ("frames not valid")
                    continue

                colour_image = np.asanyarray(colour_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                try:
                    depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)
                    point_cloud, points = self.generate_pointcloud(depth_frame, color_intrin)
                    ros_colour_img = self.bridge.cv2_to_imgmsg(colour_image, "bgr8")
                    ros_depth_img = self.bridge.cv2_to_imgmsg(depth_image, "passthrough")
                    rgb_info_msg = self.get_rgb_info(ros_colour_img.header)
                    depth_info_msg = self.get_depth_info(ros_depth_img.header)
                    point_cloud.header.stamp = ros_colour_img.header.stamp = ros_depth_img.header.stamp = \
                        rgb_info_msg.header.stamp = depth_info_msg.header.stamp = rospy.Time.now()

                    self.rgb_camera_info_pub.publish(rgb_info_msg)
                    self.img_pub.publish(ros_colour_img)
                    self.depth_camera_info_pub.publish(depth_info_msg)
                    self.depth_pub.publish(ros_depth_img)
                    self.pointcloud_pub.publish(point_cloud)
                    self.points_CF_pub.publish(points)



                except CvBridgeError:
                    pass
        finally:
            self.pipeline.stop()


def main():
    rospy.init_node('realsense_camera', anonymous=False)
    rc = RealsenseCamera()
    rc.start()

if __name__ == '__main__':
    main()