import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks

# Define motion names
name_motion = ['Walking', 'Jogging', 'Crouch']
name_grf = ['Walking_FP', 'Jogging_FP', 'Crouch_FP']

# Set the motion index
index = 1 # Example: index=0 for Walking, index=1 for Jogging, index=2 for Crouch

# Read data
data_trc = pd.read_csv(f"../Proj1files/{name_motion[index]}.csv")

# right_walking_trc = data_trc.iloc[44:153]  # Selects rows from index 46 to 150 (inclusive)
# left_walking_trc = data_trc.iloc[96:205] 
jogging_trc = data_trc.iloc[39:125]  # Selects rows from index 46 to 150 (inclusive)

data_grf = pd.read_csv(f"../Proj1files/{name_grf[index]}.csv")

# Downsample ground reaction data to match the trajectory data length
data_grf = data_grf.iloc[::10, :].reset_index(drop=True)


# Conversion factor from mm to meters
to_meters = 1 / 1000

#constants for momentum calculations
bodyWeight = 72.8
footWeight = 0.0145 * bodyWeight
gravity = 9.81
# Extract and convert relevant columns from marker trajectory data
RTOE_x = data_trc['RTOO_Y'] * to_meters
RTOE_y = data_trc['RTOO_Z'] * to_meters
LTOE_x = data_trc['LTOO_Y'] * to_meters
LTOE_y = data_trc['LTOO_Z'] * to_meters

RANKLE_x = data_trc['RAJC_Y'] * to_meters
RANKLE_y = data_trc['RAJC_Z'] * to_meters
LANKLE_x = data_trc['LAJC_Y'] * to_meters
LANKLE_y = data_trc['LAJC_Z'] * to_meters

RKNEE_x = data_trc['RKJC_Y'] * to_meters
RKNEE_y = data_trc['RKJC_Z'] * to_meters
LKNEE_x = data_trc['LKJC_Y'] * to_meters
LKNEE_y = data_trc['LKJC_Z'] * to_meters

RHIP_x = data_trc['RHJC_Y'] * to_meters
RHIP_y = data_trc['RHJC_Z'] * to_meters
LHIP_x = data_trc['LHJC_Y'] * to_meters
LHIP_y = data_trc['LHJC_Z'] * to_meters

RPELO_x = data_trc['PELO_Y'] * to_meters
RPELO_y = data_trc['PELO_Z'] * to_meters
LPELO_x = data_trc['PELO_Y'] * to_meters
LPELO_y = data_trc['PELO_Z'] * to_meters
    
RPELP_x = data_trc['PELP_Y'] * to_meters
RPELP_y = data_trc['PELP_Z'] * to_meters
LPELP_x = data_trc['PELP_Y'] * to_meters
LPELP_y = data_trc['PELP_Z'] * to_meters

RTRXO_x = data_trc['TRXO_Y'] * to_meters
RTRXO_y = data_trc['TRXO_Z'] * to_meters
LTRXO_x = data_trc['TRXO_Y'] * to_meters
LTRXO_y = data_trc['TRXO_Z'] * to_meters
    
RTRXP_x = data_trc['TRXP_Y'] * to_meters
RTRXP_y = data_trc['TRXP_Z'] * to_meters
LTRXP_x = data_trc['TRXP_Y'] * to_meters
LTRXP_y = data_trc['TRXP_Z'] * to_meters

FP1_force_x = data_grf['FP1_Force_Y']
FP1_force_y = data_grf['FP1_Force_Z']
FP1_COP_x = data_grf['FP1_COP_Y'] * to_meters
FP1_COP_y = data_grf['FP1_COP_Z'] * to_meters

FP2_force_x = data_grf['FP2_Force_Y']
FP2_force_y = data_grf['FP2_Force_Z']

FP2_COP_x = data_grf['FP2_COP_Y'] * to_meters
FP2_COP_y = data_grf['FP2_COP_Z'] * to_meters

# Trunk angle
rtrunk = np.column_stack((RTRXO_x - RTRXP_x, RTRXO_y - RTRXP_y))  # Trunk vector
rtrunk_angle_radians = np.arctan2(rtrunk[:, 0], rtrunk[:, 1])
right_trunk_angle_degrees = np.degrees(rtrunk_angle_radians)

ltrunk = np.column_stack((LTRXO_x - LTRXP_x, LTRXO_y - LTRXP_y))  # Trunk vector
ltrunk_angle_radians = np.arctan2(ltrunk[:, 0], ltrunk[:, 1])
left_trunk_angle_degrees = np.degrees(ltrunk_angle_radians)

# Left pelvis angle
lpelvis = np.column_stack((LPELP_x - LPELO_x, LPELP_y - LPELO_y))  # Left pelvis vector
lpelvis_angle_radians = np.arctan2(lpelvis[:, 0], lpelvis[:, 1])
left_pelvis_angle_degrees = np.degrees(lpelvis_angle_radians)

# Pelvis angle
rpelvis = np.column_stack((RPELP_x - RPELO_x, RPELP_y - RPELO_y))  # Pelvis vector
rpelvis_angle_radians = np.arctan2(rpelvis[:, 0], rpelvis[:, 1])
right_pelvis_angle_degrees = np.degrees(rpelvis_angle_radians)

# Left limb angles
thigh_L = np.column_stack((LHIP_x - LKNEE_x, LHIP_y - LKNEE_y))
dot_product_hip_L = np.sum(lpelvis * thigh_L, axis=1)
magnitude_lpelvis = np.linalg.norm(lpelvis, axis=1)
magnitude_thigh_L = np.linalg.norm(thigh_L, axis=1)
cos_hip_angle_L = dot_product_hip_L / (magnitude_lpelvis * magnitude_thigh_L)
hip_angle_L_radians = np.arccos(cos_hip_angle_L)
cross_product_hip_L = lpelvis[:, 0] * thigh_L[:, 1] - lpelvis[:, 1] * thigh_L[:, 0]
hip_angle_L_radians_sign = hip_angle_L_radians * np.sign(cross_product_hip_L)
hip_angle_L_degrees = np.degrees(hip_angle_L_radians_sign)

# Right limb angles
thigh_R = np.column_stack((RHIP_x - RKNEE_x, RHIP_y - RKNEE_y))
dot_product_hip_R = np.sum(rpelvis * thigh_R, axis=1)
magnitude_rpelvis = np.linalg.norm(rpelvis, axis=1)
magnitude_thigh_R = np.linalg.norm(thigh_R, axis=1)
cos_hip_angle_R = dot_product_hip_R / (magnitude_rpelvis * magnitude_thigh_R)
hip_angle_R_radians = np.arccos(cos_hip_angle_R)
cross_product_hip_R = rpelvis[:, 0] * thigh_R[:, 1] - rpelvis[:, 1] * thigh_R[:, 0]
hip_angle_R_radians_sign = hip_angle_R_radians * np.sign(cross_product_hip_R)
hip_angle_R_degrees = np.degrees(hip_angle_R_radians_sign)

# Knee angles
shank_L = np.column_stack((LKNEE_x - LANKLE_x, LKNEE_y - LANKLE_y))
dot_product_knee_L = np.sum(shank_L * thigh_L, axis=1)
magnitude_shank_L = np.linalg.norm(shank_L, axis=1)
cos_knee_angle_L = dot_product_knee_L / (magnitude_shank_L * magnitude_thigh_L)
knee_angle_L_radians = np.arccos(cos_knee_angle_L)
cross_product_knee_L = shank_L[:, 0] * thigh_L[:, 1] - shank_L[:, 1] * thigh_L[:, 0]
knee_angle_L_radians_sign = knee_angle_L_radians * np.sign(cross_product_knee_L)
knee_angle_L_degrees = np.degrees(knee_angle_L_radians_sign)

shank_R = np.column_stack((RKNEE_x - RANKLE_x, RKNEE_y - RANKLE_y))
dot_product_knee_R = np.sum(shank_R * thigh_R, axis=1)
magnitude_shank_R = np.linalg.norm(shank_R, axis=1)
cos_knee_angle_R = dot_product_knee_R / (magnitude_shank_R * magnitude_thigh_R)
knee_angle_R_radians = np.arccos(cos_knee_angle_R)
cross_product_knee_R = shank_R[:, 0] * thigh_R[:, 1] - shank_R[:, 1] * thigh_R[:, 0]
knee_angle_R_radians_sign = knee_angle_R_radians * np.sign(cross_product_knee_R)
knee_angle_R_degrees = np.degrees(knee_angle_R_radians_sign)

# Ankle angles
foot_L = np.column_stack((LANKLE_x - LTOE_x, LANKLE_y - LTOE_y))
dot_product_ankle_L = np.sum(shank_L * foot_L, axis=1)
magnitude_foot_L = np.linalg.norm(foot_L, axis=1)
cos_ankle_angle_L = dot_product_ankle_L / (magnitude_shank_L * magnitude_foot_L)
ankle_angle_L_radians = np.arccos(cos_ankle_angle_L)
cross_product_ankle_L = shank_L[:, 0] * foot_L[:, 1] - shank_L[:, 1] * foot_L[:, 0]
ankle_angle_L_radians_sign = ankle_angle_L_radians * np.sign(cross_product_ankle_L) + np.radians(5 - 90)
ankle_angle_L_degrees = np.degrees(ankle_angle_L_radians_sign)

foot_R = np.column_stack((RANKLE_x - RTOE_x, RANKLE_y - RTOE_y))
dot_product_ankle_R = np.sum(shank_R * foot_R, axis=1)
magnitude_foot_R = np.linalg.norm(foot_R, axis=1)
cos_ankle_angle_R = dot_product_ankle_R / (magnitude_shank_R * magnitude_foot_R)
ankle_angle_R_radians = np.arccos(cos_ankle_angle_R)
cross_product_ankle_R = shank_R[:, 0] * foot_R[:, 1] - shank_R[:, 1] * foot_R[:, 0]
ankle_angle_R_radians_sign = ankle_angle_R_radians * np.sign(cross_product_ankle_R) + np.radians(10 - 90)
ankle_angle_R_degrees = np.degrees(ankle_angle_R_radians_sign)

import numpy as np
import matplotlib.pyplot as plt

# Function for plotting angles with manually set y-axis limits
def plot_joint_angles(time, left_angle, right_angle, title, ylabel, ylim):
    plt.figure()
    plt.plot(time, left_angle, 'r', linewidth=3, label='Left')
    plt.plot(time, right_angle, 'b', linewidth=3, label='Right')
    plt.xlabel('Time [%]')
    plt.ylabel(ylabel)
    plt.axhline(0, linestyle='--', color='k', linewidth=1.5)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(ylim)  # Set y-axis limits
    plt.savefig("../plots/JointAngles/"+title+".png")

# Function for plotting jogging angles with manually set y-axis limits
def plot_jog_joint_angles(time, right_angle, title, ylabel, ylim):
    plt.figure()
    plt.plot(time, right_angle, 'b', linewidth=3, label='Right')
    plt.xlabel('Time [%]')
    plt.ylabel(ylabel)
    plt.axhline(0, linestyle='--', color='k', linewidth=1.5)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(ylim)  # Set y-axis limits
    plt.savefig("../plots/JoggingJointAngles/"+title+".png")

# Define y-axis limits for each joint
joint_ylim = {
    'Walking Trunk Angle': (0, 15),
    'Walking Pelvis Angle': (0, 25),
    'Walking Hip Joint Angle': (-20, 40),
    'Walking Knee Joint Angle': (-10, 80),
    'Walking Ankle Joint Angle': (-30, 20),
    'Jogging Trunk Angle': (0, 15),
    'Jogging Pelvis Angle': (0, 25),
    'Jogging Hip Joint Angle': (-20, 40),
    'Jogging Knee Joint Angle': (-10, 80),
    'Jogging Ankle Joint Angle': (-30, 20)
}

# Generate normalized time scale for plots
if index == 0:
    # Plot each joint's angles walking
    time = np.linspace(0, 100, len(right_trunk_angle_degrees[44:153]))
    plot_joint_angles(time, left_trunk_angle_degrees[95:204], right_trunk_angle_degrees[44:153], 'Walking Trunk Angle', 'Tilt [deg]', joint_ylim['Walking Trunk Angle'])
    plot_joint_angles(time, left_pelvis_angle_degrees[95:204], right_pelvis_angle_degrees[44:153], 'Walking Pelvis Angle', 'Tilt [deg]', joint_ylim['Walking Pelvis Angle'])
    plot_joint_angles(time, hip_angle_L_degrees[95:204], hip_angle_R_degrees[44:153], 'Walking Hip Joint Angle', '(-)Extension / (+)Flexion [deg]', joint_ylim['Walking Hip Joint Angle'])
    plot_joint_angles(time, knee_angle_L_degrees[95:204], knee_angle_R_degrees[44:153], 'Walking Knee Joint Angle', '(-)Extension / (+)Flexion [deg]', joint_ylim['Walking Knee Joint Angle'])
    plot_joint_angles(time, ankle_angle_L_degrees[95:204], ankle_angle_R_degrees[44:153], 'Walking Ankle Joint Angle', '(-)Plantar / (+)Dorsiflexion [deg]', joint_ylim['Walking Ankle Joint Angle'])
else:
    # Plot each joint's angles jogging
    time = np.linspace(0, 100, len(right_trunk_angle_degrees[39:125]))
    plot_jog_joint_angles(time, right_trunk_angle_degrees[39:125], 'Jogging Trunk Angle', 'Tilt [deg]', joint_ylim['Jogging Trunk Angle'])
    plot_jog_joint_angles(time, right_pelvis_angle_degrees[39:125], 'Jogging Pelvis Angle', 'Tilt [deg]', joint_ylim['Jogging Pelvis Angle'])
    plot_jog_joint_angles(time, hip_angle_R_degrees[39:125], 'Jogging Hip Joint Angle', '(-)Extension / (+)Flexion [deg]', joint_ylim['Jogging Hip Joint Angle'])
    plot_jog_joint_angles(time, knee_angle_R_degrees[39:125], 'Jogging Knee Joint Angle', '(-)Extension / (+)Flexion [deg]', joint_ylim['Jogging Knee Joint Angle'])
    plot_jog_joint_angles(time, ankle_angle_R_degrees[39:125], 'Jogging Ankle Joint Angle', '(-)Plantar / (+)Dorsiflexion [deg]', joint_ylim['Jogging Ankle Joint Angle'])

# Constants
body_weight = 72.8
g = 9.81
dt = 0.01

# 1. Hip - Knee
m1 = 0.1 * body_weight
thigh_L_length = np.sqrt(thigh_L[:, 0]**2 + thigh_L[:, 1]**2)
CoM1_xL = LHIP_x + 0.433 * (LKNEE_x - LHIP_x)
CoM1_yL = LHIP_y + 0.433 * (LKNEE_y - LHIP_y)
CoM1_vel_xL = np.gradient(CoM1_xL, dt)
CoM1_vel_yL = np.gradient(CoM1_yL, dt)
CoM1_acc_xL = np.gradient(CoM1_vel_xL, dt)
CoM1_acc_yL = np.gradient(CoM1_vel_yL, dt)
inertia1_L = ((0.323 * thigh_L_length)**2) * m1
ang_vel1_L = np.gradient(hip_angle_L_radians_sign, dt)
ang_acc1_L = np.gradient(ang_vel1_L, dt)

# 2. Knee - Ankle
m2 = 0.0465 * body_weight
shank_L_length = np.sqrt(shank_L[:, 0]**2 + shank_L[:, 1]**2)
CoM2_xL = LKNEE_x + 0.433 * (LANKLE_x - LKNEE_x)
CoM2_yL = LKNEE_y + 0.433 * (LANKLE_y - LKNEE_y)
CoM2_vel_xL = np.gradient(CoM2_xL, dt)
CoM2_vel_yL = np.gradient(CoM2_yL, dt)
CoM2_acc_xL = np.gradient(CoM2_vel_xL, dt)
CoM2_acc_yL = np.gradient(CoM2_vel_yL, dt)
inertia2_L = ((0.302 * shank_L_length)**2) * m2
ang_vel2_L = np.gradient(knee_angle_L_radians_sign, dt)
ang_acc2_L = np.gradient(ang_vel2_L, dt)

# 3. Ankle - Foot
m3 = 0.0145 * body_weight
foot_L_length = np.sqrt(foot_L[:, 0]**2 + foot_L[:, 1]**2)
CoM3_xL = LANKLE_x + 0.5 * (LTOE_x - LANKLE_x)
CoM3_yL = LANKLE_y + 0.5 * (LTOE_y - LANKLE_y)
CoM3_vel_xL = np.gradient(CoM3_xL, dt)
CoM3_vel_yL = np.gradient(CoM3_yL, dt)
CoM3_acc_xL = np.gradient(CoM3_vel_xL, dt)
CoM3_acc_yL = np.gradient(CoM3_vel_yL, dt)
inertia3_L = ((0.475 * foot_L_length)**2) * m3
ang_vel3_L = np.gradient(ankle_angle_L_radians_sign, dt)
ang_acc3_L = np.gradient(ang_vel3_L, dt)

# Moments arms
HCoM1_xL = CoM1_xL - LHIP_x
HCoM1_yL = LHIP_y - CoM1_yL
KCoM1_xL = LKNEE_x - CoM1_xL
KCoM1_yL = CoM1_yL - LKNEE_y
KCoM2_xL = LKNEE_x - CoM2_xL
KCoM2_yL = LKNEE_y - CoM2_yL
ACoM2_xL = CoM2_xL - LANKLE_x
ACoM2_yL = CoM2_yL - LANKLE_y
ACoM3_xL = CoM3_xL - LANKLE_x
ACoM3_yL = LANKLE_y - CoM3_yL
FCoM3_xL = FP1_COP_x - CoM3_xL
FCoM3_yL = CoM3_yL - FP1_COP_y

HCoM1_yL = np.array(HCoM1_yL)
HCoM1_xL = np.array(HCoM1_xL)
KCoM1_yL = np.array(KCoM1_yL)
KCoM1_xL = np.array(KCoM1_xL)
KCoM2_yL = np.array(KCoM2_yL)
KCoM2_xL = np.array(KCoM2_xL)
ACoM2_yL = np.array(ACoM2_yL)
ACoM2_xL = np.array(ACoM2_xL)
ACoM3_yL = np.array(ACoM3_yL)
ACoM3_xL = np.array(ACoM3_xL)
FP1_force_x = np.array(FP1_force_x)
CoM3_acc_xL = np.array(CoM3_acc_xL)
FCoM3_xL = np.array(FCoM3_xL)
FCoM3_yL = np.array(FCoM3_yL)
FP1_force_y = np.array(FP1_force_y)
FP1_force_x = np.array(FP1_force_x)
FP2_force_y = np.array(FP2_force_y)
FP2_force_x = np.array(FP2_force_x)

# Computation of matrix
numFramesL = len(HCoM1_xL)

# Assuming numFramesL is defined
M_L = np.zeros((9, 9, numFramesL))
Result_L = np.zeros((9, numFramesL))

g = 9.81  # Acceleration due to gravity, if needed

for i in range(numFramesL):
    M_L[:, :, i] = np.array([
        [1, 0, -1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, -1, 0, 0, 0, 0, 0],
        [HCoM1_yL[i], HCoM1_xL[i], KCoM1_yL[i], KCoM1_xL[i], 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, -1, 0, 0, 0],
        [0, 0, -KCoM2_yL[i], KCoM2_xL[i], ACoM2_yL[i], ACoM2_xL[i], 0, 1, 1],
        [0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, -ACoM3_yL[i], ACoM3_xL[i], 0, 0, 1]
    ])

for i in range(numFramesL):
    M_current_L = M_L[:, :, i]
    
    coef_current_L = np.array([
        m1 * CoM1_acc_xL[i],
        m1 * (CoM1_acc_yL[i] + g),
        inertia1_L[i] * ang_acc1_L[i],
        m2 * CoM2_acc_xL[i],
        m2 * (CoM2_acc_yL[i] + g),
        inertia2_L[i] * ang_acc2_L[i],
        m3 * CoM3_acc_xL[i] - FP1_force_x[i],
        m3 * (CoM3_acc_yL[i] + g) - FP1_force_y[i],
        FP1_force_y[i] * FCoM3_xL[i] + FP1_force_x[i] * FCoM3_yL[i] + inertia3_L[i] * ang_acc3_L[i]
    ])
    
    Result_L[:, i] = np.linalg.solve(M_current_L, coef_current_L)

# Thigh segment calculations
thigh_R_length = np.sqrt(thigh_R[:, 0]**2 + thigh_R[:, 1]**2)
CoM1_xR = RHIP_x + 0.433 * (RKNEE_x - RHIP_x)
CoM1_yR = RHIP_y + 0.433 * (RKNEE_y - RHIP_y)
CoM1_vel_xR = np.gradient(CoM1_xR, dt)
CoM1_vel_yR = np.gradient(CoM1_yR, dt)
CoM1_acc_xR = np.gradient(CoM1_vel_xR, dt)
CoM1_acc_yR = np.gradient(CoM1_vel_yR, dt)
inertia1_R = ((0.323 * thigh_R_length) ** 2) * m1
ang_vel1_R = np.gradient(hip_angle_R_radians_sign, dt)
ang_acc1_R = np.gradient(ang_vel1_R, dt)

# Shank segment calculations
shank_R_length = np.sqrt(shank_R[:, 0]**2 + shank_R[:, 1]**2)
CoM2_xR = RKNEE_x + 0.433 * (RANKLE_x - RKNEE_x)
CoM2_yR = RKNEE_y + 0.433 * (RANKLE_y - RKNEE_y)
CoM2_vel_xR = np.gradient(CoM2_xR, dt)
CoM2_vel_yR = np.gradient(CoM2_yR, dt)
CoM2_acc_xR = np.gradient(CoM2_vel_xR, dt)
CoM2_acc_yR = np.gradient(CoM2_vel_yR, dt)
inertia2_R = ((0.302 * shank_R_length) ** 2) * m2
ang_vel2_R = np.gradient(knee_angle_R_radians_sign, dt)
ang_acc2_R = np.gradient(ang_vel2_R, dt)

# Foot segment calculations
foot_R_length = np.sqrt(foot_R[:, 0]**2 + foot_R[:, 1]**2)
CoM3_xR = RANKLE_x + 0.5 * (RTOE_x - RANKLE_x)
CoM3_yR = RANKLE_y + 0.5 * (RTOE_y - RANKLE_y)
CoM3_vel_xR = np.gradient(CoM3_xR, dt)
CoM3_vel_yR = np.gradient(CoM3_yR, dt)
CoM3_acc_xR = np.gradient(CoM3_vel_xR, dt)
CoM3_acc_yR = np.gradient(CoM3_vel_yR, dt)
inertia3_R = ((0.475 * foot_R_length) ** 2) * m3
ang_vel3_R = np.gradient(ankle_angle_R_radians_sign, dt)
ang_acc3_R = np.gradient(ang_vel3_R, dt)

# Moment arms calculations
HCoM1_xR = CoM1_xR - RHIP_x
HCoM1_yR = RHIP_y - CoM1_yR
KCoM1_xR = RKNEE_x - CoM1_xR
KCoM1_yR = CoM1_yR - RKNEE_y
KCoM2_xR = RKNEE_x - CoM2_xR
KCoM2_yR = RKNEE_y - CoM2_yR
ACoM2_xR = CoM2_xR - RANKLE_x
ACoM2_yR = CoM2_yR - RANKLE_y
ACoM3_xR = CoM3_xR - RANKLE_x
ACoM3_yR = RANKLE_y - CoM3_yR
FCoM3_xR = FP2_COP_x - CoM3_xR
FCoM3_yR = CoM3_yR - FP2_COP_y

numFramesR = len(HCoM1_xR)
M_R = np.zeros((9, 9, numFramesR))
Result_R = np.zeros((9, numFramesR))



HCoM1_yR = np.array(HCoM1_yR)
HCoM1_xR = np.array(HCoM1_xR)
KCoM1_yR = np.array(KCoM1_yR)
KCoM1_xR = np.array(KCoM1_xR)
KCoM2_yR = np.array(KCoM2_yR)

KCoM2_xR = np.array(KCoM2_xR)
ACoM2_yR = np.array(ACoM2_yR)
ACoM2_xR = np.array(ACoM2_xR)
KCoM1_xR = np.array(KCoM1_xR)
ACoM3_yR = np.array(ACoM3_yR)
ACoM3_xR = np.array(ACoM3_xR)
FCoM3_xR = np.array(FCoM3_xR)
FCoM3_yR = np.array(FCoM3_yR)


g = 9.81  # Gravity acceleration if needed

for i in range(numFramesR):
    M_R[:, :, i] = np.array([
        [1, 0, -1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, -1, 0, 0, 0, 0, 0],
        [HCoM1_yR[i], HCoM1_xR[i], KCoM1_yR[i], KCoM1_xR[i], 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, -1, 0, 0, 0],
        [0, 0, -KCoM2_yR[i], KCoM2_xR[i], ACoM2_yR[i], ACoM2_xR[i], 0, 1, 1],
        [0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, -ACoM3_yR[i], ACoM3_xR[i], 0, 0, 1]
    ])

for i in range(numFramesR):
    M_current_R = M_R[:, :, i]
    
    coef_current_R = np.array([
        m1 * CoM1_acc_xR[i],
        m1 * (CoM1_acc_yR[i] + g),
        inertia1_R[i] * ang_acc1_R[i],
        m2 * CoM2_acc_xR[i],
        m2 * (CoM2_acc_yR[i] + g),
        inertia2_R[i] * ang_acc2_R[i],
        m3 * CoM3_acc_xR[i] - FP2_force_x[i],
        m3 * (CoM3_acc_yR[i] + g) - FP2_force_y[i],
        FP2_force_y[i] * FCoM3_xR[i] + FP2_force_x[i] * FCoM3_yR[i] + inertia3_R[i] * ang_acc3_R[i]
    ])
    
    Result_R[:, i] = np.linalg.solve(M_current_R, coef_current_R)

# Function for plotting joint moments and powers with manually set y-axis limits
def plot_joint_moments(normalized_time, dataL, dataR, ylabel, title, ylim):
    plt.figure()
    plt.plot(normalized_time, dataL, 'r', linewidth=3, label='Left')
    plt.plot(normalized_time, dataR, 'b', linewidth=3, label='Right')
    plt.xlabel('Time [%]')
    plt.ylabel(ylabel)
    plt.axhline(0, linestyle='--', color='k', linewidth=1.5)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(ylim)  # Set y-axis limits
    plt.savefig("../plots/Moments/"+title+".png")
    # Function for plotting joint moments and powers with manually set y-axis limits

def plot_joint_powers(normalized_time, dataL, dataR, ylabel, title, ylim):
    plt.figure()
    plt.plot(normalized_time, dataL, 'r', linewidth=3, label='Left')
    plt.plot(normalized_time, dataR, 'b', linewidth=3, label='Right')
    plt.xlabel('Time [%]')
    plt.ylabel(ylabel)
    plt.axhline(0, linestyle='--', color='k', linewidth=1.5)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(ylim)  # Set y-axis limits
    plt.savefig("../plots/Powers/"+title+".png")


# Function for plotting jogging joint moments and powers with manually set y-axis limits
def plot_jog_joint_moments(normalized_time, dataR, ylabel, title, ylim):
    plt.figure()
    plt.plot(normalized_time, dataR, 'b', linewidth=3, label='Right')
    plt.xlabel('Time [%]')
    plt.ylabel(ylabel)
    plt.axhline(0, linestyle='--', color='k', linewidth=1.5)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(ylim)  # Set y-axis limits
    plt.savefig("../plots/JogMoments/"+title+".png")


# Function for plotting jogging joint moments and powers with manually set y-axis limits
def plot_jog_joint_powers(normalized_time, dataR, ylabel, title, ylim):
    plt.figure()
    plt.plot(normalized_time, dataR, 'b', linewidth=3, label='Right')
    plt.xlabel('Time [%]')
    plt.ylabel(ylabel)
    plt.axhline(0, linestyle='--', color='k', linewidth=1.5)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(ylim)  # Set y-axis limits
    plt.savefig("../plots/JogPowers/"+title+".png")


# Define y-axis limits for each joint moment and power
joint_moment_power_ylim = {
    'Walking Hip Joint Moment': (-1.5, 1.5),
    'Walking Knee Joint Moment': (-1.5, 1.5),
    'Walking Ankle Joint Moment': (-1, 4),
    'Walking Hip Joint Power': (-5, 5),
    'Walking Knee Joint Power': (-5, 5),
    'Walking Ankle Joint Power': (-10, 15)
}

if (index == 0):
    # Normal walking


    # Time normalization
    time_points_L = len(hip_angle_L_degrees[95:204])
    normalized_time_L = np.linspace(0, 100, time_points_L)

    time_points_R = len(hip_angle_R_degrees[44:153])
    normalized_time_R = np.linspace(0, 100, time_points_R)

    # Joint Moments (L and R)
    M1_L = Result_L[6, :][95:204] / body_weight
    M1_R = Result_R[6, :][44:153] / body_weight
    M2_L = Result_L[7, :][95:204] / body_weight
    M2_R = Result_R[7, :][44:153] / body_weight
    M3_L = Result_L[8, :][95:204] / body_weight
    M3_R = Result_R[8, :][44:153] / body_weight

    # Joint Power Calculations
    total_hip_power_L = -(M1_L * ang_vel1_L[95:204]) 
    total_hip_power_R = -(M1_R * ang_vel1_R[44:153]) 
    total_knee_power_L = -(M2_L * ang_vel2_L[95:204]) 
    total_knee_power_R = -(M2_R * ang_vel2_R[44:153])
    total_ankle_power_L = -(M3_L * ang_vel3_L[95:204]) 
    total_ankle_power_R = -(M3_R * ang_vel3_R[44:153]) 


    # Plot Joint Moments
    plot_joint_moments(normalized_time_L, M1_L, M1_R, '(-)Flexion / (+)Extension [Nm/kg]', 'Walking Hip Joint Moment', joint_moment_power_ylim['Walking Hip Joint Moment'])
    plot_joint_moments(normalized_time_L, M2_L, M2_R, '(-)Flexion / (+)Extension [Nm/kg]', 'Walking Knee Joint Moment', joint_moment_power_ylim['Walking Knee Joint Moment'])
    plot_joint_moments(normalized_time_L, M3_L, M3_R, '(-)Dorsi / (+)Plantarflexion [Nm/kg]', 'Walking Ankle Joint Moment', joint_moment_power_ylim['Walking Ankle Joint Moment'])

    # Plot Joint Powers
    plot_joint_powers(normalized_time_L, total_hip_power_L, total_hip_power_R, '(-)Absorption / (+)Generation [W/kg]', 'Walking Hip Joint Power', joint_moment_power_ylim['Walking Hip Joint Power'])
    plot_joint_powers(normalized_time_L, total_knee_power_L, total_knee_power_R, '(-)Absorption / (+)Generation [W/kg]', 'Walking Knee Joint Power', joint_moment_power_ylim['Walking Knee Joint Power'])
    plot_joint_powers(normalized_time_L, total_ankle_power_L, total_ankle_power_R, '(-)Absorption / (+)Generation [W/kg]', 'Walking Ankle Joint Power', joint_moment_power_ylim['Walking Ankle Joint Power'])

        
else:
    #Jogging
 # Normal walking
    time_points_R = len(hip_angle_R_degrees[39:125])
    normalized_time_R = np.linspace(0, 100, time_points_R)

    # Joint Moments (L and R)
    M1_R = Result_R[6, :][39:125]/ body_weight
    M2_R = Result_R[7, :][39:125] / body_weight
    M3_R = Result_R[8, :][39:125] / body_weight

    # Joint Power Calculations
    total_hip_power_R = - (M1_R * ang_vel1_R[39:125]) 
    total_knee_power_R = -(M2_R * ang_vel2_R[39:125])
    total_ankle_power_R = -(M3_R * ang_vel3_R[39:125]) 

    # Creating grouped subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot Joint Moments
    plot_jog_joint_moments(normalized_time_R,  M1_R, '(-)Flexion / (+)Extension [Nm/kg]', 'Jogging Hip Joint Moment',joint_moment_power_ylim['Walking Hip Joint Moment'])
    plot_jog_joint_moments(normalized_time_R, M2_R, '(-)Flexion / (+)Extension [Nm/kg]', 'Jogging Knee Joint Moment',joint_moment_power_ylim['Walking Knee Joint Moment'])
    plot_jog_joint_moments( normalized_time_R, M3_R, '(-)Dorsi / (+)Plantarflexion [Nm/kg]', 'Jogging Ankle Joint Moment',joint_moment_power_ylim['Walking Ankle Joint Moment'])

    # Plot Joint Powers
    plot_jog_joint_powers(normalized_time_R, total_hip_power_R, '(-)Absorption / (+)Generation [W/kg]', 'Jogging Hip Joint Power',joint_moment_power_ylim['Walking Hip Joint Power'])
    plot_jog_joint_powers(normalized_time_R, total_knee_power_R, '(-)Absorption / (+)Generation [W/kg]', 'Jogging Knee Joint Power',joint_moment_power_ylim['Walking Knee Joint Power'])
    plot_jog_joint_powers(normalized_time_R, total_ankle_power_R, '(-)Absorption / (+)Generation [W/kg]', 'Jogging Ankle Joint Power',joint_moment_power_ylim['Walking Ankle Joint Power'])
