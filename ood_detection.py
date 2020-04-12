#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Code for classifying in-distribution and out-of-distribution images
based on the spectral signatures of CNN feature maps.


@author: davood
"""



####    Inspect

from __future__ import division

import numpy as np
import os
import tensorflow as tf
# import tensorlayer as tl
#from os import listdir
#from os.path import isfile, join, isdir
#import scipy.io as sio
#from skimage import io
#import skimage
#import SimpleITK as sitk
#import matplotlib.pyplot as plt
import h5py
#import scipy
#from scipy.spatial import ConvexHull
#from mpl_toolkits.mplot3d import Axes3D
#from scipy.spatial import Delaunay
import os.path
#import pandas as pd
#import sys
#import pickle
#from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
#import dk_seg
import dk_model
import dk_aux
#from PIL import Image
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import matplotlib.pyplot as plt
#from medpy.metric import hd95
from scipy import linalg
from scipy.ndimage import zoom








###   initialize the network model and restore the trained model weights

n_channel = 1
n_class = 2

gpu_ind = 1

n_feat_0 = 14
depth = 4

SX, SY, SZ= 135, 189, 155
LX, LY, LZ = 96, 96, 96

ks_0 = 3

X = tf.placeholder("float32", [None, LX, LY, LZ, n_channel])
Y = tf.placeholder("float32", [None, LX, LY, LZ, n_class])

p_keep_conv = tf.placeholder("float")
learning_rate = tf.placeholder("float")

#logit_f, _ = dk_model.davood_net(X, ks_0, depth, n_feat_0, n_channel, n_class, p_keep_conv, bias_init=0.001)
logit_f, _ , f_1, f_2= dk_model.davood_net_return_fmaps(X, ks_0, depth, n_feat_0, n_channel, n_class, p_keep_conv, bias_init=0.001)
predicter = tf.nn.softmax(logit_f)

cost= dk_model.cost_dice_forground(Y, predicter, loss_type= 'sorensen', smooth = 1e-5)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)

saver = tf.train.Saver(max_to_keep=50)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#      addressed of the saved checkpoint
restore_model_path= '/media/nerossd2/segment_everything/One_organ_all_1/models/model_saved_24_8963.ckpt'

saver.restore(sess, restore_model_path)









##  load the data to be tested.  This includes in-distribution and out-of distribution
#   datd



#######################    SPLEEN    ##########################################

spleen_dir = '/media/nerossd2/segment_everything/spleen/'

data_file_name= 'spleen_data.h5'

if os.path.exists(spleen_dir + data_file_name):

    print('Reading data')

    h5f = h5py.File(spleen_dir + data_file_name, 'r')
    X_vol_s = h5f['X_vol'][:]
    Y_vol_s = h5f['Y_vol'][:]
    info_vol_s = h5f['info_vol'][:]
    h5f.close()

    print('Finished reading data')

#######################    HEART     ##########################################

heart_dir = '/media/nerossd2/segment_everything/heart/'

data_file_name= 'heart_data.h5'

if os.path.exists(heart_dir + data_file_name):

    print('Reading data')

    h5f = h5py.File(heart_dir + data_file_name, 'r')
    X_vol_h = h5f['X_vol'][:]
    Y_vol_h = h5f['Y_vol'][:]
    info_vol_h = h5f['info_vol'][:]
    h5f.close()

    print('Finished reading data')

#######################    PROSTATE     #######################################

prostate_dir = '/media/nerossd2/segment_everything/prostate/'

data_file_name= 'prostate_data.h5'

if os.path.exists(prostate_dir + data_file_name):
    
    print('Reading data')
    
    h5f = h5py.File(prostate_dir + data_file_name, 'r')
    X_vol_p = h5f['X_vol'][:]
    Y_vol_p = h5f['Y_vol'][:]
    info_vol_p = h5f['info_vol'][:]
    h5f.close()
    
    print('Finished reading data')

#######################    HIPPOCAMPUS    #####################################

hippocampus_dir = '/media/nerossd2/segment_everything/hippocampus/'

data_file_name= 'hippocampus_data.h5'

if os.path.exists(hippocampus_dir + data_file_name):
    
    print('Reading data')
    
    h5f = h5py.File(hippocampus_dir + data_file_name, 'r')
    X_vol_c = h5f['X_vol'][:]
    Y_vol_c = h5f['Y_vol'][:]
    info_vol_c = h5f['info_vol'][:]
    h5f.close()
    
    print('Finished reading data')


#######################     pancreas         ##################################

pancreas_dir = '/media/nerossd2/segment_everything/Pancreas/'

data_file_name= 'pancreas_data_decath.h5'

if os.path.exists(pancreas_dir + data_file_name):
    
    print('Reading data')
    
    h5f = h5py.File(pancreas_dir + data_file_name, 'r')
    X_vol_r = h5f['X_vol'][:]
    Y_vol_r = h5f['Y_vol'][:]
    info_vol_r = h5f['info_vol'][:]
    h5f.close()
    
    print('Finished reading data')
    


#######################    Liver CT       #####################################

Silver_dir = '/media/nerossd2/segment_everything/Silver/'

data_file_name= 'Silver_data.h5'

if os.path.exists(Silver_dir + data_file_name):
    
    print('Reading data')
    
    h5f = h5py.File(Silver_dir + data_file_name, 'r')
    X_vol_sil = h5f['X_vol'][:]
    Y_vol_sil = h5f['Y_vol'][:]
    info_vol_sil = h5f['info_vol'][:]
    h5f.close()
    
    print('Finished reading data')


#######################    Liver-MRI-SPIR    ##################################

CHAOS_dir = '/media/nerossd2/segment_everything/CHAOS/CHAOS_Train_Sets/Train_Sets/MR/'

data_file_name= 'CHAOS_data.h5'

if os.path.exists(CHAOS_dir + data_file_name):
    
    print('Reading data')
    
    h5f = h5py.File(CHAOS_dir + data_file_name, 'r')
    X_vol_chaos = h5f['X_vol'][:]
    Y_vol_chaos = h5f['Y_vol'][:]
    info_vol_chaos = h5f['info_vol'][:]
    h5f.close()
    
    print('Finished reading data')
    

##########    Liver-MRI-DUAL-in and Liver-MRI-DUAL-out   ######################

CHAOS_dir = '/media/nerossd2/segment_everything/CHAOS/CHAOS_Train_Sets/Train_Sets/MR/'

data_file_name= 'CHAOS_data_dual.h5'

if os.path.exists(CHAOS_dir + data_file_name):
    
    print('Reading data')
    
    h5f = h5py.File(CHAOS_dir + data_file_name, 'r')
    X_vol_chaos_in   = h5f['X_vol_in'][:]
    X_vol_chaos_out  = h5f['X_vol_out'][:]
    Y_vol_chaos_in   = h5f['Y_vol_in'][:]
    Y_vol_chaos_out  = h5f['Y_vol_out'][:]
    info_vol_chaos_dual = h5f['info_vol'][:]
    h5f.close()
    
    print('Finished reading data')
    






#####    CP- newborn

DHCP_dir = '/media/nerossd2/segment_everything/DHCP/'

data_file_name= 'DHCP_data.h5'

if os.path.exists(DHCP_dir + data_file_name):

    print('Reading data')

    h5f = h5py.File(DHCP_dir + data_file_name, 'r')
    X_vol_dhcp = h5f['X_vol'][:]
    Y_vol_dhcp = h5f['Y_vol'][:]
    info_vol_dhcp = h5f['info_vol'][:]
    h5f.close()

    print('Finished reading data')









###############################################################################
##   divide the data into train, validation, and test sets
#   we do it randomly but setting the seed to get the same results.
###############################################################################


test_frac= 0.30
valid_frac= 0.40


#########  CP- fetus

data_file_name= 'CPSP_seg_data_' + str(SX) + '_' + str(SY) + '_' + str(SZ)+'.h5'

h5f = h5py.File('/media/davood/bd794557-342d-4d50-8b7f-f6c209248f89/davood/CpSpS/data/' + data_file_name, 'r')
X_vol_CP = h5f['X_vol'][:]
Y_vol_CP = h5f['Y_vol'][:]
info_vol_CP = h5f['info_vol'][:]
h5f.close()

Y_vol_CP[:,:,:,:,0]= 1- Y_vol_CP[:,:,:,:,1]
Y_vol_CP = Y_vol_CP [:, :, :, :, :2]

p_young = np.where(info_vol_CP[:, 19] > 0)[0]
p_old   = np.where(info_vol_CP[:, 19] == 0)[0]
np.random.seed(0)
np.random.shuffle(p_young)
np.random.shuffle(p_old)

p_test_yng = p_young[:int(test_frac * len(p_young))]
p_valid_yng= p_young[int(test_frac * len(p_young)):int(valid_frac * len(p_young))]
p_train_yng= p_young[int(valid_frac * len(p_young)):]

p_test_old = p_old[:int(test_frac * len(p_old))]
p_valid_old= p_old[int(test_frac * len(p_old)):int(valid_frac * len(p_old))]
p_train_old= p_old[int(valid_frac * len(p_old)):]

X_test_CP_yng = X_vol_CP[p_test_yng].copy()
X_train_CP_yng= X_vol_CP[p_train_yng].copy()
X_valid_CP_yng= X_vol_CP[p_valid_yng].copy()

Y_test_CP_yng = Y_vol_CP[p_test_yng].copy()
Y_train_CP_yng= Y_vol_CP[p_train_yng].copy()
Y_valid_CP_yng= Y_vol_CP[p_valid_yng].copy()

X_test_CP_old = X_vol_CP[p_test_old].copy()
X_train_CP_old= X_vol_CP[p_train_old].copy()
X_valid_CP_old= X_vol_CP[p_valid_old].copy()

Y_test_CP_old = Y_vol_CP[p_test_old].copy()
Y_train_CP_old= Y_vol_CP[p_train_old].copy()
Y_valid_CP_old= Y_vol_CP[p_valid_old].copy()

X_vol_CP= Y_vol_CP= 0

#########  SPLEEN

n_cases= X_vol_s.shape[0]

np.random.seed(0)
p= np.arange(n_cases)
np.random.shuffle(p)

p_test = p[:int(test_frac * n_cases)]
p_valid = p[int(test_frac * n_cases):int(valid_frac * n_cases)]
p_train= p[int(valid_frac * n_cases):]

X_train_s= X_vol_s[p_train,:,:,:,:].copy()
Y_train_s= Y_vol_s[p_train,:,:,:,:].copy()
X_test_s=  X_vol_s[p_test,:,:,:,:].copy()
Y_test_s=  Y_vol_s[p_test,:,:,:,:].copy()
X_valid_s=  X_vol_s[p_valid,:,:,:,:].copy()
Y_valid_s=  Y_vol_s[p_valid,:,:,:,:].copy()

X_vol_s= Y_vol_s= 0

#########  HEART

n_cases= X_vol_h.shape[0]

np.random.seed(0)
p= np.arange(n_cases)
np.random.shuffle(p)

p_test = p[:int(test_frac * n_cases)]
p_valid = p[int(test_frac * n_cases):int(valid_frac * n_cases)]
p_train= p[int(valid_frac * n_cases):]

X_train_h= X_vol_h[p_train,:,:,:,:].copy()
Y_train_h= Y_vol_h[p_train,:,:,:,:].copy()
X_test_h=  X_vol_h[p_test,:,:,:,:].copy()
Y_test_h=  Y_vol_h[p_test,:,:,:,:].copy()
X_valid_h=  X_vol_h[p_valid,:,:,:,:].copy()
Y_valid_h=  Y_vol_h[p_valid,:,:,:,:].copy()

X_vol_h= Y_vol_h= 0

#########  PROSTATE

n_cases= X_vol_p.shape[0]

np.random.seed(0)
p= np.arange(n_cases)
np.random.shuffle(p)

p_test = p[:int(test_frac * n_cases)]
p_valid = p[int(test_frac * n_cases):int(valid_frac * n_cases)]
p_train= p[int(valid_frac * n_cases):]

X_train_p= X_vol_p[p_train,:,:,:,0:1].copy()
Y_train_p= Y_vol_p[p_train,:,:,:,:].copy()
X_test_p=  X_vol_p[p_test,:,:,:,0:1].copy()
Y_test_p=  Y_vol_p[p_test,:,:,:,:].copy()
X_valid_p= X_vol_p[p_valid,:,:,:,0:1].copy()
Y_valid_p= Y_vol_p[p_valid,:,:,:,:].copy()

X_vol_p= Y_vol_p= 0

#########  HIPPOCAMPUS

n_cases= X_vol_c.shape[0]

Y_vol_c[:,:,:,:,1:2]= Y_vol_c[:,:,:,:,1:2] + Y_vol_c[:,:,:,:,2:]
Y_vol_c= Y_vol_c[:,:,:,:,:2]

np.random.seed(0)
p= np.arange(n_cases)
np.random.shuffle(p)

p_test = p[:int(test_frac * n_cases)]
p_valid = p[int(test_frac * n_cases):int(valid_frac * n_cases)]
p_train= p[int(valid_frac * n_cases):]

X_train_c= X_vol_c[p_train,:,:,:,0:1].copy()
Y_train_c= Y_vol_c[p_train,:,:,:,:].copy()
X_test_c=  X_vol_c[p_test,:,:,:,0:1].copy()
Y_test_c=  Y_vol_c[p_test,:,:,:,:].copy()
X_valid_c= X_vol_c[p_valid,:,:,:,0:1].copy()
Y_valid_c= Y_vol_c[p_valid,:,:,:,:].copy()

X_vol_c= Y_vol_c= 0




##########  Pancreas

n_cases= X_vol_r.shape[0]

np.random.seed(0)
p= np.arange(n_cases)
np.random.shuffle(p)

p_test = p[:int(test_frac * n_cases)]
p_valid = p[int(test_frac * n_cases):int(valid_frac * n_cases)]
p_train= p[int(valid_frac * n_cases):]

X_train_r= X_vol_r[p_train,:,:,:,:].copy()
Y_train_r= Y_vol_r[p_train,:,:,:,:].copy()
X_test_r=  X_vol_r[p_test,:,:,:,:].copy()
Y_test_r=  Y_vol_r[p_test,:,:,:,:].copy()
X_valid_r= X_vol_r[p_valid,:,:,:,:].copy()
Y_valid_r= Y_vol_r[p_valid,:,:,:,:].copy()

X_vol_r= Y_vol_r= 0




#########  Liver CT

n_cases= X_vol_sil.shape[0]

np.random.seed(0)
p= np.arange(n_cases)
np.random.shuffle(p)

p_test = p[:int(test_frac * n_cases)]
p_valid = p[int(test_frac * n_cases):int(valid_frac * n_cases)]
p_train= p[int(valid_frac * n_cases):]

X_train_sil= X_vol_sil[p_train,:,:,:,0:1].copy()
Y_train_sil= Y_vol_sil[p_train,:,:,:,:].copy()
X_test_sil=  X_vol_sil[p_test,:,:,:,0:1].copy()
Y_test_sil=  Y_vol_sil[p_test,:,:,:,:].copy()
X_valid_sil= X_vol_sil[p_valid,:,:,:,0:1].copy()
Y_valid_sil= Y_vol_sil[p_valid,:,:,:,:].copy()

X_vol_sil= Y_vol_sil= 0


#########  Liver-MRI-SPIR

n_cases= X_vol_chaos.shape[0]

np.random.seed(0)
p= np.arange(n_cases)
np.random.shuffle(p)

p_test = p[:int(test_frac * n_cases)]
p_valid = p[int(test_frac * n_cases):int(valid_frac * n_cases)]
p_train= p[int(valid_frac * n_cases):]

X_train_chaos= X_vol_chaos[p_train,:,:,:,0:1].copy()
Y_train_chaos= Y_vol_chaos[p_train,:,:,:,:].copy()
X_test_chaos=  X_vol_chaos[p_test,:,:,:,0:1].copy()
Y_test_chaos=  Y_vol_chaos[p_test,:,:,:,:].copy()
X_valid_chaos= X_vol_chaos[p_valid,:,:,:,0:1].copy()
Y_valid_chaos= Y_vol_chaos[p_valid,:,:,:,:].copy()

X_vol_chaos= Y_vol_chaos= 0

#########  Liver-MRI-DUAL-in and Liver-MRI-DUAL-out

n_cases= X_vol_chaos_in.shape[0]

np.random.seed(0)
p= np.arange(n_cases)
np.random.shuffle(p)

p_test = p[:int(test_frac * n_cases)]
p_valid = p[int(test_frac * n_cases):int(valid_frac * n_cases)]
p_train= p[int(valid_frac * n_cases):]

X_train_chaos_in= X_vol_chaos_in[p_train,:,:,:,0:1].copy()
Y_train_chaos_in= Y_vol_chaos_in[p_train,:,:,:,:].copy()
X_test_chaos_in=  X_vol_chaos_in[p_test,:,:,:,0:1].copy()
Y_test_chaos_in=  Y_vol_chaos_in[p_test,:,:,:,:].copy()
X_valid_chaos_in= X_vol_chaos_in[p_valid,:,:,:,0:1].copy()
Y_valid_chaos_in= Y_vol_chaos_in[p_valid,:,:,:,:].copy()

X_vol_chaos_in= Y_vol_chaos_in= 0

n_cases= X_vol_chaos_out.shape[0]

np.random.seed(0)
p= np.arange(n_cases)
np.random.shuffle(p)

p_test = p[:int(test_frac * n_cases)]
p_valid = p[int(test_frac * n_cases):int(valid_frac * n_cases)]
p_train= p[int(valid_frac * n_cases):]

X_train_chaos_out= X_vol_chaos_out[p_train,:,:,:,0:1].copy()
Y_train_chaos_out= Y_vol_chaos_out[p_train,:,:,:,:].copy()
X_test_chaos_out=  X_vol_chaos_out[p_test,:,:,:,0:1].copy()
Y_test_chaos_out=  Y_vol_chaos_out[p_test,:,:,:,:].copy()
X_valid_chaos_out= X_vol_chaos_out[p_valid,:,:,:,0:1].copy()
Y_valid_chaos_out= Y_vol_chaos_out[p_valid,:,:,:,:].copy()

X_vol_chaos_out= Y_vol_chaos_out= 0

#########  DHCP

X_vol_dhcp_copy= X_vol_dhcp.copy()
Y_vol_dhcp_copy= Y_vol_dhcp.copy()

n_cases= X_vol_dhcp.shape[0]

np.random.seed(0)
p= np.arange(n_cases)
np.random.shuffle(p)

p_test = p[:int(test_frac * n_cases)]
p_valid = p[int(test_frac * n_cases):int(valid_frac * n_cases)]
p_train= p[int(valid_frac * n_cases):]

X_train_dhcp= X_vol_dhcp[p_train,:,:,:,:].copy()
Y_train_dhcp= Y_vol_dhcp[p_train,:,:,:,:].copy()
X_test_dhcp=  X_vol_dhcp[p_test,:,:,:,:].copy()
Y_test_dhcp=  Y_vol_dhcp[p_test,:,:,:,:].copy()
X_valid_dhcp= X_vol_dhcp[p_valid,:,:,:,:].copy()
Y_valid_dhcp= Y_vol_dhcp[p_valid,:,:,:,:].copy()










#   Specify training and test sets

SXX, SYY, SZZ= 128, 160, 144

X_train= [X_train_CP_yng[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), X_train_CP_old[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), 
          X_train_h[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), X_train_p[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), 
          X_train_sil[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), X_train_chaos[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(),
          X_train_chaos_in[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy()]

Y_train= [Y_train_CP_yng[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), Y_train_CP_old[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), 
          Y_train_h[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), Y_train_p[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), 
          Y_train_sil[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), Y_train_chaos[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), 
          Y_train_chaos_in[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy()]

X_test= [X_test_CP_yng[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), X_test_CP_old[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), 
         X_test_h[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), X_test_p[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), 
          X_test_sil[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), X_test_chaos[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), 
          X_test_chaos_in[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), X_test_c[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), 
          X_test_r[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy()]

Y_test= [Y_test_CP_yng[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), Y_test_CP_old[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), 
         Y_test_h[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), Y_test_p[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), 
          Y_test_sil[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), Y_test_chaos[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), 
          Y_test_chaos_in[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), Y_test_c[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy(), 
          Y_test_r[:, 7:7+SXX, 10:10+SYY, 7:7+SZZ, :].copy()]

SX, SY, SZ= 128, 160, 144










LX, LY, LZ = 96, 96, 96
test_shift= LX//4
lx_list= np.squeeze( np.concatenate(  (np.arange(0, SX-LX, test_shift)[:,np.newaxis] , np.array([SX-LX])[:,np.newaxis] )  ) .astype(np.int) )
ly_list= np.squeeze( np.concatenate(  (np.arange(0, SY-LY, test_shift)[:,np.newaxis] , np.array([SY-LY])[:,np.newaxis] )  ) .astype(np.int) )
lz_list= np.squeeze( np.concatenate(  (np.arange(0, SZ-LZ, test_shift)[:,np.newaxis] , np.array([SZ-LZ])[:,np.newaxis] )  ) .astype(np.int) )
LXc, LYc, LZc= LX//4, LY//4, LZ//4

keep_test= 1.0




















#  Compute SVD of feature maps  -
#     - we actually only work with the feature number "3", which is the 
#       last (deepest) layer.

F_train_1= list()
F_train_2= list()
F_train_3= list()
F_train_4= list()
F_train_5= list()
F_train_6= list()

for i in range(len(X_train)):
    
    X_temp= X_train[i]
    Y_temp= Y_train[i]
    
    Sv_1= np.zeros( (X_temp.shape[0],n_feat_0) )
    Sv_2= np.zeros( (X_temp.shape[0],n_feat_0*2) )
    Sv_3= np.zeros( (X_temp.shape[0],n_feat_0*4) )
    Sv_4= np.zeros( (X_temp.shape[0],n_feat_0) )
    Sv_5= np.zeros( (X_temp.shape[0],n_feat_0*2) )
    Sv_6= np.zeros( (X_temp.shape[0],n_feat_0*4) )
    
    for i_c in tqdm(range(X_temp.shape[0]), ascii=True):
        
        y_sum = np.zeros((SX, SY, SZ, n_class))
        
        f_sum_1 = np.zeros((SX, SY, SZ, n_feat_0))
        f_sum_2 = np.zeros((SX//2, SY//2, SZ//2, n_feat_0*2))
        f_sum_3 = np.zeros((SX//4, SY//4, SZ//4, n_feat_0*4))
        f_sum_4 = np.zeros((SX, SY, SZ, n_feat_0))
        f_sum_5 = np.zeros((SX//2, SY//2, SZ//2, n_feat_0*2))
        f_sum_6 = np.zeros((SX//4, SY//4, SZ//4, n_feat_0*4))
        
        f_cnt = np.zeros((SX, SY, SZ))
        
        for lx in lx_list:
            for ly in ly_list:
                for lz in lz_list:
                    
                    if np.max(Y_temp[i_c:i_c + 1, lx + LXc:lx + LX - LXc, ly + LYc:ly + LY - LYc, lz + LZc:lz + LZ - LZc, 1:]) > 0:
                        
                        batch_x = X_temp[i_c:i_c + 1, lx:lx + LX, ly:ly + LY, lz:lz + LZ, :].copy()
                        
                        y_prd, f_1c, f_2c= sess.run([predicter, f_1, f_2], feed_dict={X: batch_x, p_keep_conv: 1.0})
                        
                        y_sum[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += y_prd[0, :, :, :, :]
                        
                        z= f_1c[0]
                        f_sum_1[lx:lx + LX, ly:ly + LY, lz:lz + LZ,:] += z[0, :, :, :, :].copy()
                        z= f_1c[1]
                        f_sum_2[lx//2:lx//2 + LX//2, ly//2:ly//2 + LY//2, lz//2:lz//2 + LZ//2,:] += z[0, :, :, :, :].copy()
                        z= f_1c[2]
                        f_sum_3[lx//4:lx//4 + LX//4, ly//4:ly//4 + LY//4, lz//4:lz//4 + LZ//4,:] += z[0, :, :, :, :].copy()
                        z= f_2c[0]
                        f_sum_4[lx:lx + LX, ly:ly + LY, lz:lz + LZ,:] += z[0, :, :, :, n_feat_0:].copy()
                        z= f_2c[1]
                        f_sum_5[lx//2:lx//2 + LX//2, ly//2:ly//2 + LY//2, lz//2:lz//2 + LZ//2,:] += z[0, :, :, :, 2*n_feat_0:].copy()
                        z= f_2c[2]
                        f_sum_6[lx//4:lx//4 + LX//4, ly//4:ly//4 + LY//4, lz//4:lz//4 + LZ//4,:] += z[0, :, :, :, 4*n_feat_0:].copy()
                        
                        f_cnt[lx:lx + LX, ly:ly + LY, lz:lz + LZ] += 1
                        
        y_sum = np.argmax(y_sum, axis=-1)
        y_sum[f_cnt == 0] = 0
        
        batch_x = X_temp[i_c:i_c + 1, :, :, :, :].copy()
        y_t_c=    Y_temp[i_c, :, :, :, 1].copy()
        #dk_aux.save_pred_thumbs(batch_x[0,:,:,:,0], y_t_c, y_sum, False, i, i_c, thumbs_dir )
        
        f_cnt_d = zoom( f_cnt , (0.5, 0.5, 0.5))
        f_cnt_dd= zoom( f_cnt , (0.25, 0.25, 0.25))
        
        ####  SVDs
        
        f_mean_1 = np.zeros((SXX//4, SYY//4, SZZ//4, n_feat_0))
        for i_f in range(n_feat_0):
            temp= f_sum_1[:,:,:,i_f]/(f_cnt+1e-10)
            temp= zoom( temp , (0.25, 0.25, 0.25))
            f_mean_1[:,:,:,i_f]= temp
        f_mean_1= f_mean_1.reshape((  (SX//4)*(SY//4)*(SZ//4), n_feat_0))
        U, s, Vh = linalg.svd(f_mean_1)
        Sv_1[i_c,:]= s
        
        f_mean_2 = np.zeros((SXX//4, SYY//4, SZZ//4, 2*n_feat_0))
        for i_f in range(2*n_feat_0):
            temp= f_sum_2[:,:,:,i_f]/(f_cnt_d+1e-10)
            temp= zoom( temp , (0.5, 0.5, 0.5))
            f_mean_2[:,:,:,i_f]= temp
        f_mean_2= f_mean_2.reshape((  (SX//4)*(SY//4)*(SZ//4), 2*n_feat_0))
        U, s, Vh = linalg.svd(f_mean_2)
        Sv_2[i_c,:]= s
        
        f_mean_3 = np.zeros((SXX//4, SYY//4, SZZ//4, 4*n_feat_0))
        for i_f in range(4*n_feat_0):
            temp= f_sum_3[:,:,:,i_f]/(f_cnt_dd+1e-10)
            f_mean_3[:,:,:,i_f]= temp
        f_mean_3= f_mean_3.reshape((  (SX//4)*(SY//4)*(SZ//4), 4*n_feat_0))
        U, s, Vh = linalg.svd(f_mean_3)
        Sv_3[i_c,:]= s
        
        f_mean_4 = np.zeros((SXX//4, SYY//4, SZZ//4, n_feat_0))
        for i_f in range(n_feat_0):
            temp= f_sum_4[:,:,:,i_f]/(f_cnt+1e-10)
            temp= zoom( temp , (0.25, 0.25, 0.25))
            f_mean_4[:,:,:,i_f]= temp
        f_mean_4= f_mean_4.reshape((  (SX//4)*(SY//4)*(SZ//4), n_feat_0))
        U, s, Vh = linalg.svd(f_mean_4)
        Sv_4[i_c,:]= s
        
        f_mean_5 = np.zeros((SXX//4, SYY//4, SZZ//4, 2*n_feat_0))
        for i_f in range(2*n_feat_0):
            temp= f_sum_5[:,:,:,i_f]/(f_cnt_d+1e-10)
            temp= zoom( temp , (0.5, 0.5, 0.5))
            f_mean_5[:,:,:,i_f]= temp
        f_mean_5= f_mean_5.reshape((  (SX//4)*(SY//4)*(SZ//4), 2*n_feat_0))
        U, s, Vh = linalg.svd(f_mean_5)
        Sv_5[i_c,:]= s
        
        f_mean_6 = np.zeros((SXX//4, SYY//4, SZZ//4, 4*n_feat_0))
        for i_f in range(4*n_feat_0):
            temp= f_sum_6[:,:,:,i_f]/(f_cnt_dd+1e-10)
            f_mean_6[:,:,:,i_f]= temp
        f_mean_6= f_mean_6.reshape((  (SX//4)*(SY//4)*(SZ//4), 4*n_feat_0))
        U, s, Vh = linalg.svd(f_mean_6)
        Sv_6[i_c,:]= s
        
    F_train_1.append(Sv_1)
    F_train_2.append(Sv_2)
    F_train_3.append(Sv_3)
    F_train_4.append(Sv_4)
    F_train_5.append(Sv_5)
    F_train_6.append(Sv_6)

np.save('/media/nerossd2/segment_everything/F_train_1.npy', F_train_1)
np.save('/media/nerossd2/segment_everything/F_train_2.npy', F_train_2)
np.save('/media/nerossd2/segment_everything/F_train_3.npy', F_train_3)
np.save('/media/nerossd2/segment_everything/F_train_4.npy', F_train_4)
np.save('/media/nerossd2/segment_everything/F_train_5.npy', F_train_5)
np.save('/media/nerossd2/segment_everything/F_train_6.npy', F_train_6)




F_test_1= list()
F_test_2= list()
F_test_3= list()
F_test_4= list()
F_test_5= list()
F_test_6= list()

for i in range(len(X_test)):
    
    X_temp= X_test[i]
    Y_temp= Y_test[i]
    
    Sv_1= np.zeros( (X_temp.shape[0],n_feat_0) )
    Sv_2= np.zeros( (X_temp.shape[0],n_feat_0*2) )
    Sv_3= np.zeros( (X_temp.shape[0],n_feat_0*4) )
    Sv_4= np.zeros( (X_temp.shape[0],n_feat_0) )
    Sv_5= np.zeros( (X_temp.shape[0],n_feat_0*2) )
    Sv_6= np.zeros( (X_temp.shape[0],n_feat_0*4) )
    
    for i_c in tqdm(range(X_temp.shape[0]), ascii=True):
        
        y_sum = np.zeros((SX, SY, SZ, n_class))
        
        f_sum_1 = np.zeros((SX, SY, SZ, n_feat_0))
        f_sum_2 = np.zeros((SX//2, SY//2, SZ//2, n_feat_0*2))
        f_sum_3 = np.zeros((SX//4, SY//4, SZ//4, n_feat_0*4))
        f_sum_4 = np.zeros((SX, SY, SZ, n_feat_0))
        f_sum_5 = np.zeros((SX//2, SY//2, SZ//2, n_feat_0*2))
        f_sum_6 = np.zeros((SX//4, SY//4, SZ//4, n_feat_0*4))
        
        f_cnt = np.zeros((SX, SY, SZ))
        
        for lx in lx_list:
            for ly in ly_list:
                for lz in lz_list:
                    
                    if np.max(Y_temp[i_c:i_c + 1, lx + LXc:lx + LX - LXc, ly + LYc:ly + LY - LYc, lz + LZc:lz + LZ - LZc, 1:]) > 0:
                        
                        batch_x = X_temp[i_c:i_c + 1, lx:lx + LX, ly:ly + LY, lz:lz + LZ, :].copy()
                        
                        y_prd, f_1c, f_2c= sess.run([predicter, f_1, f_2], feed_dict={X: batch_x, p_keep_conv: 1.0})
                        
                        y_sum[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += y_prd[0, :, :, :, :]
                        
                        z= f_1c[0]
                        f_sum_1[lx:lx + LX, ly:ly + LY, lz:lz + LZ,:] += z[0, :, :, :, :].copy()
                        z= f_1c[1]
                        f_sum_2[lx//2:lx//2 + LX//2, ly//2:ly//2 + LY//2, lz//2:lz//2 + LZ//2,:] += z[0, :, :, :, :].copy()
                        z= f_1c[2]
                        f_sum_3[lx//4:lx//4 + LX//4, ly//4:ly//4 + LY//4, lz//4:lz//4 + LZ//4,:] += z[0, :, :, :, :].copy()
                        z= f_2c[0]
                        f_sum_4[lx:lx + LX, ly:ly + LY, lz:lz + LZ,:] += z[0, :, :, :, n_feat_0:].copy()
                        z= f_2c[1]
                        f_sum_5[lx//2:lx//2 + LX//2, ly//2:ly//2 + LY//2, lz//2:lz//2 + LZ//2,:] += z[0, :, :, :, 2*n_feat_0:].copy()
                        z= f_2c[2]
                        f_sum_6[lx//4:lx//4 + LX//4, ly//4:ly//4 + LY//4, lz//4:lz//4 + LZ//4,:] += z[0, :, :, :, 4*n_feat_0:].copy()
                        
                        f_cnt[lx:lx + LX, ly:ly + LY, lz:lz + LZ] += 1
                        
        y_sum = np.argmax(y_sum, axis=-1)
        y_sum[f_cnt == 0] = 0
        
        batch_x = X_temp[i_c:i_c + 1, :, :, :, :].copy()
        y_t_c=    Y_temp[i_c, :, :, :, 1].copy()
        #dk_aux.save_pred_thumbs(batch_x[0,:,:,:,0], y_t_c, y_sum, False, i, i_c, thumbs_dir )
        
        f_cnt_d = zoom( f_cnt , (0.5, 0.5, 0.5))
        f_cnt_dd= zoom( f_cnt , (0.25, 0.25, 0.25))
        
        ####  SVDs
        
        f_mean_1 = np.zeros((SXX//4, SYY//4, SZZ//4, n_feat_0))
        for i_f in range(n_feat_0):
            temp= f_sum_1[:,:,:,i_f]/(f_cnt+1e-10)
            temp= zoom( temp , (0.25, 0.25, 0.25))
            f_mean_1[:,:,:,i_f]= temp
        f_mean_1= f_mean_1.reshape((  (SX//4)*(SY//4)*(SZ//4), n_feat_0))
        U, s, Vh = linalg.svd(f_mean_1)
        Sv_1[i_c,:]= s
        
        f_mean_2 = np.zeros((SXX//4, SYY//4, SZZ//4, 2*n_feat_0))
        for i_f in range(2*n_feat_0):
            temp= f_sum_2[:,:,:,i_f]/(f_cnt_d+1e-10)
            temp= zoom( temp , (0.5, 0.5, 0.5))
            f_mean_2[:,:,:,i_f]= temp
        f_mean_2= f_mean_2.reshape((  (SX//4)*(SY//4)*(SZ//4), 2*n_feat_0))
        U, s, Vh = linalg.svd(f_mean_2)
        Sv_2[i_c,:]= s
        
        f_mean_3 = np.zeros((SXX//4, SYY//4, SZZ//4, 4*n_feat_0))
        for i_f in range(4*n_feat_0):
            temp= f_sum_3[:,:,:,i_f]/(f_cnt_dd+1e-10)
            f_mean_3[:,:,:,i_f]= temp
        f_mean_3= f_mean_3.reshape((  (SX//4)*(SY//4)*(SZ//4), 4*n_feat_0))
        U, s, Vh = linalg.svd(f_mean_3)
        Sv_3[i_c,:]= s
        
        f_mean_4 = np.zeros((SXX//4, SYY//4, SZZ//4, n_feat_0))
        for i_f in range(n_feat_0):
            temp= f_sum_4[:,:,:,i_f]/(f_cnt+1e-10)
            temp= zoom( temp , (0.25, 0.25, 0.25))
            f_mean_4[:,:,:,i_f]= temp
        f_mean_4= f_mean_4.reshape((  (SX//4)*(SY//4)*(SZ//4), n_feat_0))
        U, s, Vh = linalg.svd(f_mean_4)
        Sv_4[i_c,:]= s
        
        f_mean_5 = np.zeros((SXX//4, SYY//4, SZZ//4, 2*n_feat_0))
        for i_f in range(2*n_feat_0):
            temp= f_sum_5[:,:,:,i_f]/(f_cnt_d+1e-10)
            temp= zoom( temp , (0.5, 0.5, 0.5))
            f_mean_5[:,:,:,i_f]= temp
        f_mean_5= f_mean_5.reshape((  (SX//4)*(SY//4)*(SZ//4), 2*n_feat_0))
        U, s, Vh = linalg.svd(f_mean_5)
        Sv_5[i_c,:]= s
        
        f_mean_6 = np.zeros((SXX//4, SYY//4, SZZ//4, 4*n_feat_0))
        for i_f in range(4*n_feat_0):
            temp= f_sum_6[:,:,:,i_f]/(f_cnt_dd+1e-10)
            f_mean_6[:,:,:,i_f]= temp
        f_mean_6= f_mean_6.reshape((  (SX//4)*(SY//4)*(SZ//4), 4*n_feat_0))
        U, s, Vh = linalg.svd(f_mean_6)
        Sv_6[i_c,:]= s
        
    F_test_1.append(Sv_1)
    F_test_2.append(Sv_2)
    F_test_3.append(Sv_3)
    F_test_4.append(Sv_4)
    F_test_5.append(Sv_5)
    F_test_6.append(Sv_6)

np.save('/media/nerossd2/segment_everything/F_test_1.npy', F_test_1)
np.save('/media/nerossd2/segment_everything/F_test_2.npy', F_test_2)
np.save('/media/nerossd2/segment_everything/F_test_3.npy', F_test_3)
np.save('/media/nerossd2/segment_everything/F_test_4.npy', F_test_4)
np.save('/media/nerossd2/segment_everything/F_test_5.npy', F_test_5)
np.save('/media/nerossd2/segment_everything/F_test_6.npy', F_test_6)












###############################################################################
###   Load the saved spectral signatures for analysis


#  This is the index of the features maps used.  We only use number "3",
#  which is the deepest layer.
i_F= 3

features_dir= 'C:\\Users\\davoo\\Dropbox\\scratch\\thumbs_svd_mixed\\'

F_train_1= list(np.load(features_dir+'F_train_1.npy'))
F_train_2= list(np.load(features_dir+'F_train_2.npy'))
F_train_3= list(np.load(features_dir+'F_train_3.npy'))
F_train_4= list(np.load(features_dir+'F_train_4.npy'))
F_train_5= list(np.load(features_dir+'F_train_5.npy'))
F_train_6= list(np.load(features_dir+'F_train_6.npy'))

F_test_1= list(np.load(features_dir+'F_test_1.npy'))
F_test_2= list(np.load(features_dir+'F_test_2.npy'))
F_test_3= list(np.load(features_dir+'F_test_3.npy'))
F_test_4= list(np.load(features_dir+'F_test_4.npy'))
F_test_5= list(np.load(features_dir+'F_test_5.npy'))
F_test_6= list(np.load(features_dir+'F_test_6.npy'))

F_train= [F_train_1, F_train_2, F_train_3, F_train_4, F_train_5, F_train_6]
F_test = [F_test_1, F_test_2, F_test_3, F_test_4, F_test_5, F_test_6]

for i in range(len(F_train)):
    for j in range(len(F_train[i])):
        temp= F_train[i][j]
        temp= np.log(temp)
        temp= temp/np.linalg.norm(temp, axis=1)[:,np.newaxis]
        F_train[i][j]= temp

for i in range(len(F_test)):
    for j in range(len(F_test[i])):
        temp= F_test[i][j]
        temp= np.log(temp)
        temp= temp/np.linalg.norm(temp, axis=1)[:,np.newaxis]
        F_test[i][j]= temp


colors= ['b', 'g', 'r', 'c', 'm', 'y', 'k']


plt.figure()
for i in range(7):
    plt.plot(F_train[i_F][i].T, colors[i])
for i in range(7,10):
    plt.plot(F_test[i_F][i].T, colors[i-7], marker='*')




#  Here, you must specify the indices of in-distribution training data (for F_tr),
#    in-distribution test data (for F_te), and out-of-distribution data (for F_od)
    
F_tr= F_train[i_F][0]
for i in range(1,7):
    F_tr= np.concatenate((F_tr, F_train[i_F][i]), axis=0)

F_te= F_test[i_F][0]
for i in range(1,7):
    F_te= np.concatenate((F_te, F_test[i_F][i]), axis=0)

F_od= F_test[i_F][7]
for i in range(8,10):
    F_od= np.concatenate((F_od, F_test[i_F][i]), axis=0)

D_tr= np.zeros(F_tr.shape[0])
for i in range(F_tr.shape[0]):
    F_tr_reduced= np.delete(F_tr, i, axis=0)
    D_tr[i]= np.linalg.norm( F_tr[i,:]- F_tr_reduced, axis=1 ).min()

D_te= np.zeros(F_te.shape[0])
for i in range(F_te.shape[0]):
    D_te[i]= np.linalg.norm( F_te[i,:]- F_tr, axis=1 ).min()

D_od= np.zeros(F_od.shape[0])
for i in range(F_od.shape[0]):
    D_od[i]= np.linalg.norm( F_od[i,:]- F_tr, axis=1 ).min()


#  This plots the histograms

plt.figure(), 
plt.hist(D_tr, facecolor='b', bins=30), 
plt.hist(D_te, facecolor='g', bins=10), 
plt.hist(D_od, facecolor='r', bins=10)


# this computes the threshold tau for classification

tau= D_tr.mean()+D_tr.std()*2.5

# compute the confusion matrix

CM= np.zeros((2,2))
CM[0,0]= np.sum(D_te<tau)
CM[0,1]= np.sum(D_te>=tau)
CM[1,0]= np.sum(D_od<tau)
CM[1,1]= np.sum(D_od>=tau)

# print the accuracy, sensitivity, and specificity

print( (CM[0,0]+CM[1,1])/CM.sum(), CM[1,1]/(CM[1,0]+CM[1,1]), CM[0,0]/(CM[0,0]+CM[0,1]) )

# compute AUC and plot ROC

auc, fpr, tpr= dk_aux.compute_AUC(D_te, D_od, np.linspace(0,10,5000))
plt.figure(), plt.plot(fpr, tpr)














