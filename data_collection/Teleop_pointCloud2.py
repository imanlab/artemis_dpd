import numpy as np
import SLRobot
import json, codecs
import roslibpy
from scipy.spatial.transform import Rotation as R
import time
import pandas as pd
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R


def receive_sensor(msg):
    global sensor
    sensor = msg

def scale_delta(x):
    if abs(x[0]) < 0.005 and abs(x[1]) < 0.005 and abs(x[2]) < 0.005:
        x = [0,0,0]
    return x

def get_correspondence_indices(P, Q):
    """For each point in P find closest one in Q."""
    dist = np.linalg.norm((Q - P), axis=1)
    chosen_idx = dist.argsort()[:2000]

    return chosen_idx


def point_phantom(data, EE_pose):
    chosen_idx = get_correspondence_indices(EE_pose, data)
    close_point = data[chosen_idx[0]]
    return close_point, chosen_idx


if __name__ == "__main__":

    sensor = None

    client = roslibpy.Ros(host='127.0.0.1', port=9090)
    client.run()
    sub_data = roslibpy.Topic(client, '/xServTopic', 'xela_server/XStream')


    Z = np.ones(24)
    X = np.ones(24)
    Y = np.ones(24)


    # Control gains

    # pgain = 0.1 * np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0], dtype=np.float64)
    pgain = 0.1 * np.array([600.0, 600.0, 600.0, 600.0, 300.0, 300.0, 50.0], dtype=np.float64)
    # dgain = 0.2 * np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0], dtype=np.float64)
    dgain = 0.1 * np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0], dtype=np.float64)

    theta_gain = 0.01 * np.array([300.0, 300.0, 10.0], dtype=np.float64)

    spring_gain = 50



    robotSlave = SLRobot.SLRobot(robot_name='franka_s', backend_addr='tcp://192.168.4.5:51468',local_addr='192.168.4.6', gripper_actuation=False)

    robotMaster = SLRobot.SLRobot(robot_name='franka_m')

    robotSlave.use_inv_dyn = False
    robotMaster.use_inv_dyn = False



    # desired_joint_pos1 = np.array([0, 0, 0, - 1.562, 0, 1.914, 0])
    desired_joint_pos1 = np.array([-0.01954646, -0.5118466,   0.04858341, -2.42494202,  0.01152505,  1.93963933,   0.74695224])

    robotMaster.gotoJointPosition(desired_joint_pos1)
    robotSlave.gotoJointPosition(desired_joint_pos1)


    robotMaster.logger.maxTimeSteps = 10000
    robotSlave.logger.maxTimeSteps = 10000

    robotMaster.startLogging()
    robotSlave.startLogging()

    init_torque_Master = np.zeros((7,))


    Delta_Z_target = 0.005


    n = 12
    xyz_points = pd.read_csv(('/home/marta/ncnr_ws/src/ncnr/sl_lcas/sl/build/sl_panda/python/SLexamples/phantom_PC/phantom%s.csv' % (n)),header=None)
    xyz_points = np.asarray(xyz_points, np.float32)

    mat_ref = pd.read_csv(('/home/marta/ncnr_ws/src/ncnr/sl_lcas/sl/build/sl_panda/python/SLexamples/phantom_PC/rot_matrix%s.csv' % (n)),header=None)
    mat_ref = np.asarray(mat_ref, np.float32)




    data = {'Joint_pos':[], 'joint_velocity':[], 'transf_matrix':[], 'Torque':[], 'Normal':[], 'Shear_x':[], 'Shear_y':[], 'time':[]}
    start_time = time.time()

    try:
        while True:
            Z_phantom = -20
            F_spring = 0

            sub_data.subscribe(receive_sensor)



            if sensor != None:
                for i in range(24):
                    Z[i] = np.array(sensor['data'][0]['xyz'][i]['z'])
                    X[i] = np.array(sensor['data'][0]['xyz'][i]['x'])
                    Y[i] = np.array(sensor['data'][0]['xyz'][i]['y'])

            robotMaster.command = init_torque_Master

            #teleoperation control
            j_posdes_slave = robotMaster.current_j_pos
            j_veldes_slave = robotMaster.current_j_vel

            target_j_acc = pgain * (j_posdes_slave - robotSlave.current_j_pos) - dgain * robotSlave.current_j_vel
            robotSlave.command = target_j_acc

            #master control
            nearest_point, index = point_phantom(xyz_points,robotMaster.current_c_pos)
            delta_theta = [0, 0, 0]

            if np.linalg.norm((nearest_point - robotMaster.current_c_pos)) < 0.09:
                Z_phantom = xyz_points[index].mean(axis=0)[2]


                current_rot = robotMaster.o_t_ee_Matrix[0:3, 0:3]
                ref_rot = np.transpose(np.reshape(mat_ref[index][0], (3, 3)))

                delta_rot = np.dot(ref_rot, current_rot.T)
                angle = R.from_dcm(delta_rot)
                delta_theta = angle.as_euler('xyz', degrees=False)
                if np.absolute(delta_theta[0]) > 0.523599 or np.absolute(delta_theta[1]) > 0.523599:
                    delta_theta = [0, 0, 0]
                # print(angle.as_euler('xyz', degrees=False))

            torque = theta_gain * delta_theta
            # torque=[0,0,0]

            Z_sensor = robotMaster.o_t_ee_Matrix[2, 3]-0.035




            Delta_Z = Z_phantom - Z_sensor
            print(Delta_Z)

            if Delta_Z > 0 and Delta_Z <= Delta_Z_target:
               F_spring = -0.6 * Delta_Z

            if Delta_Z > Delta_Z_target:
               F_spring = -(0.6 * Delta_Z_target - 6*(np.exp(Delta_Z-Delta_Z_target) - 1))
            # F_spring=0
            F_master = [0, 0, spring_gain*F_spring, torque[0], torque[1], torque[2]]

            init_torque_Master = np.dot(robotMaster.getJacobian().T, F_master)
            # print(init_torque_Master)



            robotMaster.nextStep()
            robotSlave.nextStep()

            data["Joint_pos"].append(robotSlave.current_j_pos.tolist())
            data["joint_velocity"].append(robotSlave.current_j_vel.tolist())
            data["transf_matrix"].append(robotSlave.o_t_ee_Matrix.tolist())
            data["Torque"].append(robotSlave.current_load.tolist())
            data["Normal"].append(Z.tolist())
            data["Shear_x"].append(X.tolist())
            data["Shear_y"].append(Y.tolist())
            data["time"].append(time.time() - start_time)


    except KeyboardInterrupt:
        file_path = ("Palpation_Data/newdata%s_a.json" %(n))  ## your path variable
        json.dump(data, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True,indent=4)  ### this saves the array in .json format




        robotMaster.stopLogging()
        # robotSlave.stopLogging()

    # data_json = data.tolist()  # nested lists with same data, indices
    # file_path = "Data/data1.json"  ## your path variable
    # json.dump(data, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True,
    #           indent=4)  ### this saves the array in .json format

#
# obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
# b_new = json.loads(obj_text)
# a_new = np.array(b_new)