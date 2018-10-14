from transforms3d.euler import euler2mat,mat2euler
import p3_utils
from numpy import unravel_index
import numpy as np



def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum() # only difference

#def softmax(x):
#    """Compute softmax values for each sets of scores in x."""
#    return np.exp(x) / np.sum(np.exp(x), axis=0)

def resample(xt1, weight_t1, p_num):
     # INPUT
     # xt1: particles get from prediction step
     # weight_t1: weight get from update step
     # p_num: the total number of particles, does not change in entire process

  X_new = np.zeros((3, p_num))
  weight2 = np.tile(1/p_num, p_num)
  j = 0
  c = weight_t1[0][0] # weight_t1 is normalized
  for k in range(p_num):
    u = np.random.uniform(0, 1/p_num) #create a uniform distribution u
    beta = u + k/p_num
    #scan each part in the circle
    while beta > c :
         j = j +1
         c = c + weight_t1[0][j]
         # increasing the decision section length
    X_new[:, k] = xt1[:, j]
    #if Beta is smaller than many times, put this repeated particles j in new set

  return weight2, X_new    # resampled new_weight, and new_X particles



def pfilter(O_t, O_t1, p_num, xt, weight, b_T_l, m_scan, mt, MAP, x_range, y_range, x_im,y_im):
    # INPUT
    # ot ot1: lidar pose data
    # p_num: the total number of particles
    # weight: weight 1/number of particles
    # xt initial state
    # b_t_l lidar to body transformation
    # m_scan meansurement data from lidar scan
    # mt map
    # MAP map
    # x y range:
    # x_im y_im:   physical x,y positions of the grid map cells

    ########################################locolization prediction##########################################
  delta = O_t1 - O_t
  # number of p_num particles                                               #use 0.13 for joint0
  # generate noise                                                          #use 0.13 for joint1
  xt1 = xt + np.tile(delta, p_num)       #use 0.13 for joint3
                                                                            #use 0.05 for joint2
                                                                            #use 0.105  for testset


    ########################################locolization update##########################################

  scan = np.dot(b_T_l, m_scan)
  corrs = []
 #correlation for each particle
  for i in range(p_num):
        Ph_state = xt1[:, i]
        #particle Ph_state

        w_T_b= np.zeros((4, 4))
        Rz_particle = euler2mat(0, 0, Ph_state[2], axes='sxyz')
        w_T_b[0:3, 0:3] = Rz_particle
        w_T_b[0, 3]=Ph_state[0]
        w_T_b[1, 3]=Ph_state[1]
        w_T_b[2, 3]=0.93
        w_T_b[3, 3]=1

        Y = np.dot(w_T_b, scan)
        # converte measurement to one particle Ph_state
        temp1 = Y[2] >= 1.03
        Y = Y[:, temp1]

        corr_matrix = p3_utils.mapCorrelation(mt, x_im,y_im, Y[0:3, :], x_range, y_range)
        # find the max in the 9*9 grid
        corr = np.max(corr_matrix)
        location = unravel_index(corr_matrix.argmax(), corr_matrix.shape)

        delta_x = (location[0] - 4)*MAP['res']
        delta_y = (location[1] - 4)*MAP['res']

        xt1[0, i] = xt1[0, i] + delta_x
        xt1[1, i] = xt1[1, i] + delta_y
        corrs.append(corr)

  Ph_state = np.array(softmax(np.array(corrs)).reshape((1,p_num)))

  #update the weight here
  weight_t1 = np.array(weight * Ph_state / np.sum(weight * Ph_state)).reshape((1, p_num))

 # get the largest weight particle's index
  max_index = np.argmax(weight_t1)
  best_particle = xt1[:, max_index]
 # get the body to world transformation (current best particle)
  w_T_b1 = np.zeros((4, 4))
  Rz_particle = euler2mat(0, 0, best_particle[2], axes='sxyz')
  w_T_b1[0:3, 0:3] = Rz_particle
  w_T_b1[0, 3] =best_particle[0]
  w_T_b1[1, 3] =best_particle[1]
  w_T_b1[2, 3] =0.93
  w_T_b1[3, 3] =1

  Neff = 1/np.dot(weight_t1.reshape((1,p_num)), weight_t1.reshape((p_num,1)))

  if Neff < 5:
    weight_t1, xt1 = resample(xt1, weight_t1, p_num)

  return xt1, weight_t1, w_T_b1, best_particle



