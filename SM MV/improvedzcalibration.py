from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import glob, cv2, sys, time
from sklearn.decomposition import PCA
from general_robotics_toolbox import *

sys.path.append('toolbox')
from robot_def import *
from utils import *
from rpi_ati_net_ft import *
sys.path.append('robot_motion')
from RobotMotionController import *
from calibration import calibrate

####################################################FT Connection####################################################
# ati_tf=NET_FT('192.168.60.100')
# ati_tf.start_streaming()
#Homogenous matrix from pen tip to ati sensor
H_pentip2ati=np.loadtxt(r'C:\Users\maima\OneDrive\Documents\Python Scripts\Robot Sheet Metal/probetip2ati.csv',delimiter=',')

H_ati2pentip=np.linalg.inv(H_pentip2ati)

ad_ati2pentip=adjoint_map(Transform(H_ati2pentip[:3,:3],H_ati2pentip[:3,-1]))
ad_ati2pentip_T=ad_ati2pentip.T
#################### FT Connection ####################
#Client object that connects to server sensor
RR_ati_cli=RRN.ConnectService('rr+tcp://localhost:59823?service=ati_sensor')


#########################################################Robot config parameters#########################################################
robot=robot_obj('ABB_1200_5_90',r"C:\Users\maima\OneDrive\Documents\Python Scripts\Robot Sheet Metal\ABB_1200_5_90_robot_default_config.yml",tool_file_path=r"C:\Users\maima\OneDrive\Documents\Python Scripts\Robot Sheet Metal\rig_pen.csv")
q_seed=np.zeros(6)

#Connect to robot IP, uses external guided motion
abb_robot_ip = '192.168.60.101'
#For EGM, the freq that the robot is controlled is 4 ms --> .004 s
TIMESTEP=0.004
controller_params = {
    #cntrl damping relates to OS
    "force_ctrl_damping": 60.0, # 200, 180, 90, 60
    "force_epsilon": 0.1, # Unit: N
    "moveL_speed_lin": 6.0, # 10 Unit: mm/sec
    "moveL_acc_lin": 7.2, # Unit: mm/sec^2 0.6, 1.2, 3.6
    "moveL_speed_ang": np.radians(10), # Unit: rad/sec
    "trapzoid_slope": 1, # trapzoidal load profile. Unit: N/sec
    "load_speed": 20.0, # Unit mm/sec 10
    "unload_speed": 1.0, # Unit mm/sec
    'settling_time': .2, # Unit: sec
    "lookahead_time": 0.132, # Unit: sec, 0.02
    "jogging_speed": 50, # Unit: mm/sec
    "jogging_acc": 10, # Unit: mm/sec^2
    'force_filter_alpha': 0.9 # force low pass filter alpha
    }


#Measure the z and y displacement from the flange to the tool tip
Pft = np.array([-55.755, 0, 130.05])

tool_T = Transform(np.eye(3), Pft)

#Before running, jog end effector to four inner corners of rig
#If values aren't near these SOMETHING IS WRONG
#bottom left
c1 = np.array([547.68, 139.10, 330.14]) 
#bottom right --> Po, rig origin point
c2 = np.array([547.68, 58.30, 329.90]) 
#top left
c3 = np.array([642.33, 139.10, 326.82])
#top right
c4 = np.array([642.32, 58.31, 326.58]) 

#Quaternion from teachpendant, base to flange
RoF = q2R(np.array([.02368, -.00020, -.99971, -.00441]))

#Measure the z and x displacement from the flange to the tool tip
Pft = np.array([-55.755, 0, 130.05])

c1 = c1 + RoF@Pft
c2 = c2 + RoF@Pft
c3 = c3 + RoF@Pft
c4 = c4 + RoF@Pft

#saves raw Pbr, Rbr to .csv file, ready for second calibration for z axis
#calibrate(c1,c2,c3,c4)

rig_pose=np.loadtxt(r"C:\Users\maima\OneDrive\Documents\Python Scripts\Robot Sheet Metal\rig_pose_raw_test.csv",delimiter=',')
#rig_pose = calibrate(c1,c2,c3,c4) #Gets raw rig pose H
w = 85.2 # width of the rig (parallel to y-axis)
h = 100.07
l = 19.7
thickness = 2.84

rig_pose[:3,-1]=rig_pose[:3,-1]+rig_pose[:3,:3]@np.array([h/2,w/2,0])


R_pencil=rig_pose[:3,:3]@Ry(np.pi)

mctrl=MotionController(robot,rig_pose,H_pentip2ati,controller_params,TIMESTEP,FORCE_PROTECTION=5,RR_ati_cli=RR_ati_cli,abb_robot_ip=abb_robot_ip)

corners_offset=np.array([[h/2+l/2,0,0],[0,w/2+l/2,0],[-h/2-l/2,0,0],[0,-w/2-l/2,0]])

corners=np.dot(rig_pose[:3,:3],corners_offset.T).T+np.tile(rig_pose[:3,-1],(4,1))

###loop four corners to get precise position base on force feedback
corners_adjusted=[]
f_d=-1	#10N push down
mctrl.start_egm()
for corner in corners:
	try:
		corner_top=corner+20*rig_pose[:3,-2]
		corner_top_safe=corner+50*rig_pose[:3,-2]
		print(corner_top)
		print(corner_top_safe)
		input("Move to corner")
		q_corner_top=robot.inv(corner_top,R_pencil,q_seed)[0]	###initial joint position
		q_corner_top_safe=robot.inv(corner_top_safe,R_pencil,q_seed)[0]
		mctrl.jog_joint_position_cmd(q_corner_top_safe,v=controller_params["jogging_speed"])
		input("Push")
		mctrl.jog_joint_position_cmd(q_corner_top,v=controller_params["jogging_speed"])

		time.sleep(0.1)
		# ati_tf.set_tare_from_ft()	#clear bias
		mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool")) # clear bias
		# res, tf, status = ati_tf.try_read_ft_streaming(.1)###get force feedback
		time.sleep(0.1)
		mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool")) # clear bias
		time.sleep(0.1)
		print("Current force reading:",mctrl.ft_reading)
		input("Start pushing")

		ft_record = mctrl.force_load_z(f_d)
		
		for i in range(100): # making sure to get the latest joint position
			q_cur = mctrl.read_position()
		corners_adjusted.append(robot.fwd(q_cur).p)
		print("Adjusted corner:",corners_adjusted[-1])

		mctrl.jog_joint_position_cmd(q_corner_top_safe,v=controller_params["jogging_speed"])

		ft_record = np.array(ft_record)
		plt.plot(ft_record[:,0],ft_record[:,1],'-o')
		plt.xlabel('Time')
		plt.ylabel('Force')
		plt.show()
	except (Exception,KeyboardInterrupt) as e:
		print("Error:", e)
		mctrl.stop_egm()
		exit()

mctrl.stop_egm()


###UPDATE IPAD POSE based on new corners
p_all=np.array(corners_adjusted)
np.savetxt(r"C:\Users\maima\OneDrive\Documents\Python Scripts\Robot Sheet Metal\corners_adjusted.csv", p_all, delimiter=',')
#identify the center point and the plane
center=np.mean(p_all,axis=0)
pca = PCA()
pca.fit(p_all)
R_temp = pca.components_.T		###decreasing variance order
if R_temp[:,0]@center<0:		###correct orientation
	R_temp[:,0]=-R_temp[:,0]
if R_temp[:,-1]@R_pencil[:,-1]>0:
	R_temp[:,-1]=-R_temp[:,-1]

R_temp[:,1]=np.cross(R_temp[:,2],R_temp[:,0])
# center = center - R_temp[:,2]*thickness
# center = center - R_temp[:,0]*h/2
# center = center - R_temp[:,1]*w/2
center = center + R_temp@np.array([-h/2,-w/2,-thickness])

print("New rig R:", R_temp)
print("New rig center:", center)
# np.savetxt()
np.savetxt('rig_pose.csv', H_from_RT(R_temp,center), delimiter=',')
		