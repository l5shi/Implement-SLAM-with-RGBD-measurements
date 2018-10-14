import numpy as np
from transforms3d.euler import euler2mat
import p3_utils
from load_data import get_rgb,get_depth,get_lidar,get_joint,getRGBCalib,getIRCalib,getExtrinsics_IR_RGB



def texture_mapping( MAP, w_T_b_best):

  IRCalib = getIRCalib()
  Matrix_IRCalib = np.array([[IRCalib['fc'][0], IRCalib['ac']*IRCalib['fc'][0], IRCalib['cc'][0]],
                             [0, IRCalib['fc'][1], IRCalib['cc'][1]],
                             [0,0,1]])
  RGBCalib =getRGBCalib()
  Matrix_RGBCalib = np.array([[RGBCalib['fc'][0], RGBCalib['ac']*RGBCalib['fc'][0], RGBCalib['cc'][0]],
                             [0, RGBCalib['fc'][1], RGBCalib['cc'][1]],
                             [0,0,1]])
  exIR_RGB = getExtrinsics_IR_RGB()

  l0 = get_lidar("lidar/train_lidar0")
  r0 = get_rgb("cam/RGB_0")
  d0 = get_depth("cam/DEPTH_0")

  lidar_t = []
  depth_t = []
  Tneck = np.zeros((4, 4))
  Thead = np.zeros((4, 4))

  for i in range(len(l0)):
    lidar_t1 = np.array(l0[i]['t'][0])
    lidar_t.append(lidar_t1)

  for j in range(len(d0)):
    depth_t1 = np.array(d0[j]['t'][0])
    depth_t.append(depth_t1)


  rgb_it = [r['image'] for r in r0]
  rgb_angle = [r['head_angles'] for r in r0]
  rgb_neck = rgb_angle[0]
  rgb_head = rgb_angle[1]

  depth_dt = [d['depth'] for d in d0]

  ############## do the time alignment for lidar time compare with depth time
  xxx = np.array((lidar_t))
  yyy = np.array((depth_t))

  xx = np.array(([l[0] for l in xxx]))
  yy = np.array(([l[0] for l in yyy]))
  idx1 = np.searchsorted(xx, yy, side="left").clip(max=xx.size - 1)
  mask = (idx1 > 0) & \
         ((idx1 == len(xx)) | (np.fabs(yy - xx[idx1 - 1]) < np.fabs(yy - xx[idx1])))
  ##### the index is the resulting lidar timestamp index that has the closest time step as the depth's timestamp
  index1 = np.array((idx1 - mask)).tolist()

  #print(index)


  # transform from ir to rgb frame rgb T_ir
  rgb_T_ir = np.zeros((4, 4))
  rgb_T_ir[0:3, 0:3] = exIR_RGB['rgb_R_ir']
  rgb_T_ir[0, 3] =exIR_RGB['rgb_T_ir'][0]
  rgb_T_ir[1, 3] =exIR_RGB['rgb_T_ir'][1]
  rgb_T_ir[2, 3] =exIR_RGB['rgb_T_ir'][2]
  rgb_T_ir[3, 3] =1

  # tranform from rgb to body frame  b_T_rgb

  # use the idxs to track the
  Rn = euler2mat(0, 0, rgb_neck, axes='sxyz')
  Tneck[0:3, 0:3] = Rn
  Tneck[0, 3] =0
  Tneck[1, 3] =0
  Tneck[2, 3] =0.07
  Tneck[3, 3] =1
  Rh = euler2mat(0, rgb_head, 0, axes='sxyz')
  Thead[0:3, 0:3] = Rh    # transform from lidar to body frame
  Thead[0, 3] =0
  Thead[1, 3] =0
  Thead[2, 3] =0.33
  Thead[3, 3] =1
  b_T_rgb = np.dot(Thead, Tneck)


  # w_T_b is the current best particle
  w_T_c_rgb = np.dot(w_T_b_best, b_T_rgb)

  o_T_c = np.array([[0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]])

  map_x = MAP['sizex']
  map_y = MAP['sizey']

  my_x = map_x.tolist()
  my_y = map_y.tolist()

  tmt = np.zeros((MAP['sizex'], MAP['sizey'], 3))

  #tmt[my_x, my_y, 0] =
  #tmt[my_x, my_y, 1] =
  #tmt[my_x, my_y, 2] =

  return tmt