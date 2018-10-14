import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat, mat2euler
from load_data import get_lidar,get_joint
import math
from math import log,pi
from particle_filter import pfilter
from scipy.special import expit
from p3_utils import bresenham2D
import cv2
##################################### Read data ########################################

#  from the testmapcorr function: init MAP
MAP={}
MAP['res']=0.05 #change to 0.05 for training set
MAP['xmin']=-40 #meters
MAP['ymin']=-40                   # need to change  the map scale for dataset 2
MAP['xmax']=40
MAP['ymax']=40
MAP['sizex']=int(np.ceil((MAP['xmax']-MAP['xmin']) / MAP['res'] +1))
MAP['sizey']=int(np.ceil((MAP['ymax']-MAP['ymin']) / MAP['res'] +1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']))
# from the testmapcorr function: get the x range and y range

x_range=np.arange(-0.2,0.2+0.05,0.05)
y_range=np.arange(-0.2,0.2+0.05,0.05)  # for test set

#x_range=np.arange(-0.2,0.2+0.05,0.05)
#y_range=np.arange(-0.2,0.2+0.05,0.05)  # for training set


x_im=np.arange(MAP['xmin'],MAP['xmax']+MAP['res'], MAP['res'])
y_im=np.arange(MAP['ymin'],MAP['ymax']+MAP['res'], MAP['res'])


N=100 #change to 200 particles for training set 2 to get better map
timestep=100 # pick small value,i.e. 20 for the timestep, will get a pretty clear map but with a long time

trajectory = []

##################################### Read data ########################################

j0 = get_joint("joint/train_joint0")
l0 = get_lidar("lidar/train_lidar0")
#  r0 = get_rgb("cam/RGB_0")
#  d0 = get_depth("cam/DEPTH_0")
#  exIR_RGB = getExtrinsics_IR_RGB()
#  IRCalib = getIRCalib()
#  RGBCalib = getRGBCalib()

# visualize data
#  replay_lidar(l0)
#  replay_rgb(r0)

l0_pose = [l['pose'] for l in l0]
l0_scan = [l['scan'] for l in l0]
j0_angle= j0['head_angles']
j0_neck=j0_angle[0]
j0_head=j0_angle[1]


################################################ mapping #################################################

data1=np.array(j0['head_angles'])
rneck=np.array(data1[0])
rhead=np.array(data1[1])

i=0
j=0
k=0
lidar_t=[]

length=l0[0]['scan'].shape[1]


for i in range (len(l0)):
     j_t = np.array(j0['ts']).T
     lidar_t1=np.array(l0[i]['t'][0])
     lidar_t.append(lidar_t1)

################## find the nearest timestamp in joint comparing with lidar's timestamp##############
xxx=np.array((j_t))
yyy=np.array((lidar_t))

xx = np.array(([l[0] for l in xxx]))
yy = np.array(([l[0] for l in yyy]))
idx = np.searchsorted(xx, yy, side="left").clip(max=xx.size-1)
mask = (idx > 0) &  \
       ( (idx == len(xx)) | (np.fabs(yy - xx[idx-1]) < np.fabs(yy - xx[idx])) )
out = xx[idx-mask]
##### the out is the resulting joint timestamp that has the closest size as the lidar's timestamp
index=np.array((idx-mask)).tolist()
num=len(index)


#print(theta)
#theta=np.arange(-180,180,3.6)/180*math.pi
#theta=theta.reshape(1,N)


Xp=np.zeros((1,N))
Yp=np.zeros((1,N))
theta=np.zeros((1,N))
#initial condition [0, 0, 0]
xt=np.array([Xp,Yp,theta]).reshape(3,N)
weight=np.array([1/N]*N)


mt=MAP['map']
odd_map=np.zeros((MAP['sizex'],MAP['sizey']))
#odometry measurement
Ot=np.array([0,0,0]).reshape(3,1)
angles=np.arange(-135/180*math.pi,135/180*math.pi,0.00436332)

num1=np.arange(0,num,timestep)
for k in num1:
    para=4
    Th=np.zeros((4,4))
    Tn=np.zeros((4,4))
    w_T_l=np.zeros((4,4))

    # from the testmapcorr function: take valid indices
    ranges=np.double(l0_scan[k].squeeze())
    indValid=np.logical_and((ranges<30),(ranges>0.1))
    ranges=ranges[indValid]
    angles_2=angles[indValid]

    # from the testmapcorr function: xy position in the sensor frame
    xs1=np.array([ranges*np.cos(angles_2)])
    ys1=np.array([ranges*np.sin(angles_2)])
    # convert position in the map frame here
    scan0=np.concatenate([np.concatenate([xs1,ys1],axis=0),np.zeros(xs1.shape)],axis=0)
    scan1=np.concatenate([scan0,np.ones(xs1.shape)],axis=0)


    R_neck=euler2mat(0,0,j0_neck[index[k]],axes='sxyz')
    Tn[0:3, 0:3]=R_neck
    Tn[0,3]=0
    Tn[1,3]=0
    Tn[2,3]=0.15
    Tn[3,3]=1

    R_head=euler2mat(0,j0_head[index[k]],0,axes='sxyz')
    Th[0:3,0:3]=R_head
    Th[0, 3] =0
    Th[1, 3] =0
    Th[2, 3] =0.33
    Th[3, 3] =1

    #define lidar to body transformtion
    b_T_l=np.dot(Th,Tn)


    l0_pose_t= l0_pose[k].squeeze()
    Rw_t_l=euler2mat(0,0,l0_pose_t[2],axes='sxyz')
    w_T_l[0:3,0:3]=Rw_t_l
    w_T_l[0, 3]=l0_pose_t[0]
    w_T_l[1, 3]=l0_pose_t[1]
    w_T_l[2, 3]=1.41
    w_T_l[3, 3]=1
    #define lidar to world transformtion

    O_t1=np.dot(w_T_l,np.linalg.inv(b_T_l))
    theta=mat2euler(O_t1[0:3,0:3])

    Ot1=np.array([O_t1[0,3],O_t1[1,3],theta[2]]).reshape((3,1))

    ##################################### particle filter ########################################

    xt1, weight_t1, w_O_b_best,best = pfilter(Ot,Ot1,N,xt,weight,b_T_l,scan1,mt,MAP,x_range,y_range,x_im,y_im)

    ##################################### texture map ########################################

    #tmt=texture_mapping(MAP,w_T_V_BEST)


    ##################################### update the map ########################################

    beam = np.dot(b_T_l, scan1)
    # get Yworld
    beam1 = np.dot(w_O_b_best, beam)

    position = beam1[2] >=1.03
    beam1 = beam1[:,position]


    for i in range(beam1.shape[1]):
        # convert from meters to cells
        x_s = np.ceil((best[0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
        y_s = np.ceil((best[1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

        x_end = np.ceil((beam1[0, i] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
        y_end = np.ceil((beam1[1, i] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

        # use bresenham2D to find free cells
        p_scan = bresenham2D(x_s, y_s, x_end, y_end)
        p_scan = p_scan.astype(int)

        # mi =mi+log(b) free
        odd_map[p_scan[0][-1], p_scan[1][-1]] = odd_map[p_scan[0][-1], p_scan[1][-1]] + log(para)
        # mi =mi+log(1/b) occ
        odd_map[p_scan[0][:-1], p_scan[1][:-1]] = odd_map[p_scan[0][:-1], p_scan[1][:-1]] + log(1 / para)

    # recover the map from log odd, p(mi=1)=1-1/(1+exp(mi))
    P_occupied = 1-expit(-odd_map)

    occ = P_occupied > 0.95
    free = P_occupied < 0.05

    mt1=occ*1+free*(-1)

    xt=xt1

    weight=weight_t1 #update weight

    mt=mt1#←Mapping(zt + 1, μ∗t∣t, bTh, mt)
    trajectory.append(best)


    # μ(i)t+1∣t←LocalizationPrediction(μ(i)t∣t,ot,ot+1)
    # (μ(i)t+1∣t+1,α(i)t+1∣t+1)←LocalizationUpdate(μ(i)t+1∣t,α(i)t∣t,zt+1,mt,bTh)
    Ot=Ot1
    print(k)    #check the status of my loop


##################################### update the map with trajectory ########################################
trajectory = np.array(trajectory)
#trajectory of the best particle

mt_tra = ((1. - 1. * mt) / 2.).astype(np.float32)
mt_tra = cv2.cvtColor(mt_tra, cv2.COLOR_GRAY2RGB)

for ii in range(len(trajectory) - 1):
        x0 = np.ceil((trajectory[ii, 0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
        y0 = np.ceil((trajectory[ii, 1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

        x1 = np.ceil((trajectory[ii + 1, 0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
        y1 = np.ceil((trajectory[ii + 1, 1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1
        # draw the trajectory with cv2 line
        cv2.line(mt_tra, (y1, x1), (y0, x0), (0, 255, 0), thickness=2)
plt.imshow(mt_tra)
plt.show()






