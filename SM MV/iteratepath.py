import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
from general_robotics_toolbox import tesseract as rox_tesseract
from general_robotics_toolbox import robotraconteur as rr_rox
import abb_motion_program_exec as abb
from matplotlib import pyplot as plt
from calibration import calibrate

final_rig_pose=np.loadtxt("rig_pose.csv",delimiter=',')

Pbr = final_rig_pose[0:3,-1]

#Angle adjustment about z axis, makes parallel with rig axis
z_theta = 3.4279*np.pi/180     #Radians
#current z axis
vz = final_rig_pose[0:3, 2]
Rz = rot(vz, z_theta)

Rbr = final_rig_pose[0:3, 0:3]*Rz

Tcb = homohogenous

qbr = R2q(Rbr)

# Define the robot
with open('ABB_1200_5_90_robot_default_config.yml', 'r') as file:
    robot = rr_rox.load_robot_info_yaml_to_robot(file)

#Measure the z and y displacement from the flange to the tool tip
Pft = np.array([-55.755, 0, 130.05])

tool_T = Transform(np.eye(3), Pft)

# Run it on the robot
my_tool = abb.tooldata(True,abb.pose(tool_T.p,R2q(tool_T.R)),abb.loaddata(0.001,[0,0,0.001],[1,0,0,0],0,0,0))
my_wobj = abb.wobjdata(False,True,"",abb.pose(Pbr + [7,-5,0], qbr),abb.pose([0,0,0],[1,0,0,0]))

def quadrant(q,robot):
	cf146=np.floor(np.array([q[0],q[3],q[5]])/(np.pi/2))
	eef=fwdkin(robot,q).p
	
	REAR=(1-np.sign((rot([0,0,1],q[0])@np.array([1,0,0]))@np.array([eef[0],eef[1],eef[2]])))/2

	LOWERARM= q[2]<-np.pi/2
	FLIP= q[4]<0

	return np.hstack((cf146,[4*REAR+2*LOWERARM+FLIP])).astype(int)

mp = abb.MotionProgram(tool=my_tool,wobj=my_wobj)

### You may want to use moveabsj to move to the initial joint angles
### Provide the initial joint angles
# jt_init = abb.jointtarget(np.degrees(curve_js[0]),[0]*6) # 
# mp.MoveAbsJ(jt_init,abb.v1000,abb.fine)
###

# Rotation are all np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
#Points 0 is -1

#Check origin
p1 = [75,75,20] 

corner_p = np.array([p1])
corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
for i in range(len(corner_p)):
    robt = abb.robtarget(corner_p[i],R2q(corner_R),abb.confdata(0,-1,-1,0),[0]*6) # create the robtarget, position (mm), orientation (quaternion)
    if i==0 or i==4:
        mp.MoveL(robt,abb.v5,abb.fine) # last zone is fine
    else:
        mp.MoveL(robt,abb.v5,abb.fine) # using zone = z50

print("Robot start moving")

#client = abb.MotionProgramExecClient(base_url="http://127.0.0.1:80") # for simulation in RobotStudio
client = abb.MotionProgramExecClient(base_url="http://192.168.60.101:80") # for real robot
log_results = client.execute_motion_program(mp) # run on the robot/robotstudio and log the results
exit()

for j in range(0,13):
    #Points 0 is -1
    p1 = [20, 20, 5] 
    p2 = [20, 20, 0-.02*j]
    p3 = [20, 60, 0-.02*j]
    p4 = [20, 60, 5]

    p4 = [30, 20, 5] 
    p5 = [30, 20, 0-.02*j]
    p6 = [30, 60, 0-.02*j]
    p7 = [30, 60, 5]

    p8 = [40, 20, 5] 
    p9 = [40, 20, 0-.02*j]
    p10 = [40, 60,0-.02*j]
    p11= [40, 60, 5]

    
    corner_p = np.array([p1,p2,p3,p4, p5, p6, p7, p8, p9, p10, p11])
    corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
    for i in range(len(corner_p)):
        robt = abb.robtarget(corner_p[i],R2q(corner_R),abb.confdata(0,-1,-1,0),[0]*6) # create the robtarget, position (mm), orientation (quaternion)
        if i==0 or i==4:
            mp.MoveL(robt,abb.v5,abb.fine) # last zone is fine
        else:
            mp.MoveL(robt,abb.v5,abb.fine) # using zone = z0
    
    print("Robot start moving")
    
#client = abb.MotionProgramExecClient(base_url="http://127.0.0.1:80") # for simulation in RobotStudio
client = abb.MotionProgramExecClient(base_url="http://192.168.60.101:80") # for real robot
log_results = client.execute_motion_program(mp) # run on the robot/robotstudio and log the results

exit()

cmd_num=log_results.data[:,1] # command number
start_id = np.where(cmd_num==2)[0][0]
curve_js_exe = np.radians(log_results.data[start_id:,2:8]) # executed joint angles
curve_p_exe = [] # executed flange position
for i in range(len(curve_js_exe)):
    curve_p_exe.append(fwdkin(robot, curve_js_exe[i]).p)
lam_exe = np.cumsum(np.linalg.norm(np.diff(curve_p_exe, axis=0), axis=1)) # executed path length
lam_exe = np.insert(lam_exe, 0, 0)

# visualize the joint trajectory
fig = plt.figure()
for i in range(6):
    plt.plot(lam_exe,curve_js_exe[:,i], label=f'Joint {i+1} executed', linestyle='--')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Joint angle (rad)')
plt.title('Joint trajectory')
plt.show()
