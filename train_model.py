#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 10 2020

Code to train the CNN segmentation models.



"""



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
import pickle
#from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import dk_seg
import dk_model
import dk_aux
#from PIL import Image
#import pandas as pd
#from scipy.stats import beta
from medpy.metric import hd95, assd, asd



n_channel = 1
n_class = 2

SX, SY, SZ= 135, 189, 155




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

    with open(DHCP_dir + 'subject_tag.txt', 'rb') as f:
        subject_tag= pickle.load(f)

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














#########   Specify training and test sets

n_tr= np.array([X_train_CP_yng.shape[0], X_train_CP_old.shape[0], X_train_dhcp.shape[0], X_train_h.shape[0],
                X_train_p.shape[0], X_train_sil.shape[0], X_train_chaos.shape[0]])
sampling_mode= 'root-proportional'
if sampling_mode=='proportional':
    q= np.ones((1,1))
    for i in range(len(n_tr)):
        q= np.concatenate(( q, np.ones((n_tr[i],1))/len(n_tr)/n_tr[i] ))
    q= q[1:]
    q= np.cumsum(q)
    if np.sum(n_tr>0)==1:
        q/= q.max()
elif sampling_mode=='log-proportional':
    q= np.ones((1,1))
    for i in range(len(n_tr)):
        q= np.concatenate(( q, np.ones((n_tr[i],1))/len(n_tr)/np.log(n_tr[i]) ))
    q= q[1:]
    q= np.cumsum(q)
    q/= q.max()
elif sampling_mode=='root-proportional':
    q= np.ones((1,1))
    for i in range(len(n_tr)):
        q= np.concatenate(( q, np.ones((n_tr[i],1))/len(n_tr)/np.sqrt(n_tr[i]) ))
    q= q[1:]
    q= np.cumsum(q)
    q/= q.max()
elif sampling_mode=='4throot-proportional':
    q= np.ones((1,1))
    for i in range(len(n_tr)):
        q= np.concatenate(( q, np.ones((n_tr[i],1))/len(n_tr)/(n_tr[i]**0.25) ))
    q= q[1:]
    q= np.cumsum(q)
    q/= q.max()
elif sampling_mode=='davood-proportional':
    q= np.ones((1,1))
    for i in range(len(n_tr)):
        q= np.concatenate(( q, np.ones((n_tr[i],1))/len(n_tr)/(n_tr[i]**0.75) ))
    q= q[1:]
    q= np.cumsum(q)
    q/= q.max()
else:
    print('Sampling mode unrecognized!')


X_train= np.concatenate( (X_train_CP_yng, X_train_CP_old, X_train_dhcp, X_train_h, X_train_p, X_train_sil, X_train_chaos), axis=0 )
Y_train= np.concatenate( (Y_train_CP_yng, Y_train_CP_old, Y_train_dhcp, Y_train_h, Y_train_p, Y_train_sil, Y_train_chaos), axis=0 )
X_test = np.concatenate( (X_test_CP_yng, X_test_CP_old, X_test_s, X_test_h, X_test_p, X_test_sil, X_test_chaos, X_test_chaos_in, X_test_chaos_out), axis=0 )
Y_test = np.concatenate( (Y_test_CP_yng, Y_test_CP_old, Y_test_s, Y_test_h, Y_test_p, Y_test_sil, Y_test_chaos, Y_test_chaos_in, Y_test_chaos_out), axis=0 )
X_valid= np.concatenate( (X_valid_CP_yng, X_valid_CP_old, X_valid_s, X_valid_h, X_valid_p, X_valid_sil), axis=0 )
Y_valid= np.concatenate( (Y_valid_CP_yng, Y_valid_CP_old, Y_valid_s, Y_valid_h, Y_valid_p, Y_valid_sil), axis=0 )

#n_tr= np.array([X_train_CP.shape[0], X_train_s.shape[0], X_train_h.shape[0], X_train_p.shape[0], X_train_sil.shape[0]])
#q= np.ones((1,1))
#for i in range(len(n_tr)):
#    q= np.concatenate(( q, np.ones((n_tr[i],1))/len(n_tr)/n_tr[i] ))
#q= q[1:]
#q= np.cumsum(q)
#
#X_train= np.concatenate( (X_train_CP, X_train_s, X_train_h, X_train_p, X_train_sil), axis=0 )
#Y_train= np.concatenate( (Y_train_CP, Y_train_s, Y_train_h, Y_train_p, Y_train_sil), axis=0 )
#X_test = np.concatenate( (X_test_CP, X_test_s, X_test_h, X_test_p, X_test_sil, X_test_list), axis=0 )
#Y_test = np.concatenate( (Y_test_CP, Y_test_s, Y_test_h, Y_test_p, Y_test_sil, Y_test_list), axis=0 )
#X_valid= np.concatenate( (X_valid_CP, X_valid_s, X_valid_h, X_valid_p, X_valid_sil), axis=0 )
#Y_valid= np.concatenate( (Y_valid_CP, Y_valid_s, Y_valid_h, Y_valid_p, Y_valid_sil), axis=0 )







print(n_tr)
print(q)





assert(len(q)==X_train.shape[0])




n_train = X_train.shape[0]
n_test =  X_test.shape[0]
n_valid = X_valid.shape[0]

X_vol= Y_vol= 0

print('n_train ', n_train, n_valid, n_test)
















#########   Define the model and set the training parameters


LX, LY, LZ = 96, 96, 96
test_shift= LX//4
lx_list= np.squeeze( np.concatenate(  (np.arange(0, SX-LX, test_shift)[:,np.newaxis] , np.array([SX-LX])[:,np.newaxis] )  ) .astype(np.int) )
ly_list= np.squeeze( np.concatenate(  (np.arange(0, SY-LY, test_shift)[:,np.newaxis] , np.array([SY-LY])[:,np.newaxis] )  ) .astype(np.int) )
lz_list= np.squeeze( np.concatenate(  (np.arange(0, SZ-LZ, test_shift)[:,np.newaxis] , np.array([SZ-LZ])[:,np.newaxis] )  ) .astype(np.int) )

#LXc, LYc, LZc= LX//4, LY//4, LZ//4

n_feat_0 = 14
depth = 4

ks_0 = 3

L_Rate = 10.0e-5



n_channel= 1
n_class= 2


X = tf.placeholder("float32", [None, LX, LY, LZ, n_channel])
Y = tf.placeholder("float32", [None, LX, LY, LZ, n_class])
learning_rate = tf.placeholder("float")
p_keep_conv = tf.placeholder("float")


logit_f, _ = dk_model.davood_net(X, ks_0, depth, n_feat_0, n_channel, n_class, p_keep_conv, bias_init=0.001)

predicter = tf.nn.softmax(logit_f)


cost= dk_model.cost_dice_forground(Y, predicter, loss_type= 'sorensen', smooth = 1e-5)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)




#T_vars = tf.trainable_variables()
#fine_tune_vars = [var for var in T_vars if var.name.endswith('out:0')]
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=fine_tune_vars)
#print('\n'*3, 'ONLY these are optimized:    ', fine_tune_vars , '\n'*3)



gpu_ind= 1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)


saver = tf.train.Saver(max_to_keep=50)

sess = tf.Session()
sess.run(tf.global_variables_initializer())



#restore_model_path= '/media/nerossd2/segment_everything/thumbs_3/pretrain_dhcp/models/model_saved_3_8586.ckpt'
#restore_model_path= '/media/nerossd2/segment_everything/thumbs_3/finetune_yng/models/model_saved_9_8396.ckpt'
#saver.restore(sess, restore_model_path)



i_global = 0
best_test = 0

i_eval = -1

#ADD_NOISE = True
#EPOCH_BEGIN_NOISE = 10
#noise_sigma = 0.03


batch_size = 1
n_epochs = 150


test_interval = 2000
test_interval_uncr= 3
n_MC=10

#center_jitter= 36
center_jitter_x= (SX-LX)//2-5
center_jitter_y= (SY-LY)//2-5
center_jitter_z= (SZ-LZ)//2-5


keep_train= 0.9
keep_test= 1.0
keep_uncert= 0.9







results_dir= ''


for epoch_i in range(n_epochs):

    for i in range(n_train):
        
        im_ind= np.random.rand()
        im_ind= np.where(q>im_ind)[0][0]
        
        x_c, y_c, z_c= SX//2, SY//2, SZ//2
        
        x_c+= np.random.randint(-center_jitter_x, center_jitter_x+1)
        y_c+= np.random.randint(-center_jitter_y, center_jitter_y+1)
        z_c+= np.random.randint(-center_jitter_z, center_jitter_z+1)
        
        x_i= np.min( [ np.max( [x_c, LX//2 ] ), SX-LX//2-1 ] )- LX//2
        y_i= np.min( [ np.max( [y_c, LY//2 ] ), SY-LY//2-1 ] )- LY//2
        z_i= np.min( [ np.max( [z_c, LZ//2 ] ), SZ-LZ//2-1 ] )- LZ//2
        
        batch_x = X_train[im_ind * batch_size:(im_ind + 1) * batch_size, x_i:x_i+LX,  y_i:y_i+LY,  z_i:z_i+LZ, :].copy()
        batch_y = Y_train[im_ind * batch_size:(im_ind + 1) * batch_size, x_i:x_i+LX,  y_i:y_i+LY,  z_i:z_i+LZ, :].copy()
        
        
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, learning_rate: L_Rate, p_keep_conv: keep_train})
        batch_x = batch_y = 0
            
            
        i_global += 1
        
        
        if i_global % test_interval == 0:
            
            i_eval += 1
            
            print(epoch_i, i, i_global)
            
            
            #cost_c = np.zeros(X_train.shape[0])
            dice_c = np.zeros((X_train.shape[0], n_class+1))
            # err_tr = np.zeros((X_train.shape[0], 1))
            
            for i_c in tqdm(range(X_train.shape[0]), ascii=True):
                
                y_tr_pr_sum = np.zeros((SX, SY, SZ, n_class))
                y_tr_pr_cnt = np.zeros((SX, SY, SZ))
                
                for lx in lx_list:
                    for ly in ly_list:
                        for lz in lz_list:
                            
                            batch_x = X_train[i_c:i_c + 1, lx:lx + LX, ly:ly + LY, lz:lz + LZ, :].copy()
                            
                            pred_temp = sess.run(predicter, feed_dict={X: batch_x, p_keep_conv: keep_test})
                            y_tr_pr_sum[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += pred_temp[0, :, :, :, :]
                            y_tr_pr_cnt[lx:lx + LX, ly:ly + LY, lz:lz + LZ] += 1
                                
                y_tr_pr_c = np.argmax(y_tr_pr_sum, axis=-1)
                y_tr_pr_c[y_tr_pr_cnt == 0] = 0
                
                #batch_x = X_train[i_c:i_c + 1, :,:,:, :].copy()
                batch_y = Y_train[i_c:i_c + 1, :, :, :, :].copy()
                
                for j_c in range(n_class):
                    dice_c[i_c, j_c] = dk_seg.dice( batch_y[0, :, :, :, j_c] == 1 , y_tr_pr_c == j_c )
                    
            print('train dice   %.3f' % dice_c[:, 0].mean(), ', %.3f' % dice_c[:, 1].mean(), ', %.3f' % dice_c[:, 2].mean())
            
            
            
            #cost_c = np.zeros(X_valid.shape[0])
            dice_c = np.zeros((X_valid.shape[0], n_class+1))
            # err_tr = np.zeros((X_valid.shape[0], 1))
            
            for i_c in tqdm(range(X_valid.shape[0]), ascii=True):
                
                y_tr_pr_sum = np.zeros((SX, SY, SZ, n_class))
                y_tr_pr_cnt = np.zeros((SX, SY, SZ))
                
                for lx in lx_list:
                    for ly in ly_list:
                        for lz in lz_list:
                            
                            batch_x = X_valid[i_c:i_c + 1, lx:lx + LX, ly:ly + LY, lz:lz + LZ, :].copy()
                            
                            pred_temp = sess.run(predicter, feed_dict={X: batch_x, p_keep_conv: keep_test})
                            y_tr_pr_sum[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += pred_temp[0, :, :, :, :]
                            y_tr_pr_cnt[lx:lx + LX, ly:ly + LY, lz:lz + LZ] += 1
                                
                y_tr_pr_c = np.argmax(y_tr_pr_sum, axis=-1)
                y_tr_pr_c[y_tr_pr_cnt == 0] = 0
                
                #batch_x = X_valid[i_c:i_c + 1, :,:,:, :].copy()
                batch_y = Y_valid[i_c:i_c + 1, :, :, :, :].copy()
                
                for j_c in range(n_class):
                    dice_c[i_c, j_c] = dk_seg.dice( batch_y[0, :, :, :, j_c] == 1 , y_tr_pr_c == j_c )
                    
            print('valid dice   %.3f' % dice_c[:, 0].mean(), ', %.3f' % dice_c[:, 1].mean(), ', %.3f' % dice_c[:, 2].mean())
            
            valid_dice= dice_c[:, 1].mean()
            
            
            dice_c = np.zeros((X_test.shape[0], n_class+7))
            
            for i_c in tqdm(range(X_test.shape[0]), ascii=True):
                
                y_tr_pr_sum = np.zeros((SX, SY, SZ, n_class))
                y_tr_pr_cnt = np.zeros((SX, SY, SZ))
                
                for lx in lx_list:
                    for ly in ly_list:
                        for lz in lz_list:
                            
                            batch_x = X_test[i_c:i_c + 1, lx:lx + LX, ly:ly + LY, lz:lz + LZ, :].copy()
                            
                            pred_temp = sess.run(predicter,  feed_dict={X: batch_x, p_keep_conv: keep_test})
                            y_tr_pr_sum[lx:lx + LX, ly:ly + LY, lz:lz + LZ,:] += pred_temp[0, :, :, :, :]
                            y_tr_pr_cnt[lx:lx + LX, ly:ly + LY, lz:lz + LZ] += 1
                                
                y_tr_pr_c = np.argmax(y_tr_pr_sum, axis=-1)
                y_tr_pr_c[y_tr_pr_cnt == 0] = 0
                
                batch_x = X_test[i_c:i_c + 1, :, :, :, :].copy()
                batch_y = Y_test[i_c:i_c + 1, :, :, :, :].copy()
                
                for j_c in range(n_class):
                    dice_c[i_c, j_c] = dk_seg.dice(batch_y[0, :, :, :, j_c] == 1, y_tr_pr_c == j_c)
                    
                y_t_c = np.argmax( batch_y[0, :, :, :, :], axis=-1)
                
                #dk_aux.save_pred_thumbs(batch_x[0,:,:,:,0], y_t_c, y_tr_pr_c, False, i_c, i_eval, images_dir )
                
                '''if i_eval==0:
                    save_pred_mhds(batch_x[0,:,:,:,0], y_t_c, y_tr_pr_c, False, i_c, i_eval)
                else:
                    save_pred_mhds(None, None, y_tr_pr_c, False, i_c, i_eval)'''
                
                dice_c[i_c, n_class] = hd95(y_t_c, y_tr_pr_c)
                dice_c[i_c, n_class+1] = asd(y_t_c, y_tr_pr_c)
                dice_c[i_c, n_class+2] = assd(y_t_c, y_tr_pr_c)
                
                y_tr_pr_soft= y_tr_pr_sum[:,:,:,1]/(y_tr_pr_cnt+1e-10)
                #dk_aux.save_pred_soft_thumbs(batch_x[0,:,:,:,0], y_t_c, y_tr_pr_c, y_tr_pr_soft, False, i_c, i_eval, images_dir)
                
                error_mask= dk_aux.seg_2_anulus(y_t_c, radius= 2.0)
                
                plot_save_path= None
                ECE, MCE, ECE_curve= dk_aux.estimate_ECE_and_MCE(y_t_c, y_tr_pr_soft, plot_save_path=plot_save_path)
                dice_c[i_c, n_class+3]= ECE
                dice_c[i_c, n_class+4]= MCE
                
                plot_save_path= None
                ECE, MCE, ECE_curve= dk_aux.estimate_ECE_and_MCE_masked(y_t_c, y_tr_pr_soft, error_mask, plot_save_path=plot_save_path)
                dice_c[i_c, n_class+5]= ECE
                dice_c[i_c, n_class+6]= MCE
                
                #Y_pred_1[i_c, :,:,:]= y_tr_pr_soft.copy()
            
            
            #print('test cost   %.3f' % cost_c.mean())
            print('test dice   %.3f' % dice_c[:, 0].mean(), ', %.3f' % dice_c[:, 1].mean(), ', %.3f' % dice_c[:, 2].mean())
            
            np.savetxt(results_dir + 'stats_test_' + str(i_eval) + '.txt', dice_c, fmt='%6.3f', delimiter=',')
            
            
            if i_eval % test_interval_uncr == 0 and i_eval>0:
                
                dice_c = np.zeros((X_test.shape[0], n_class+10))

                for i_c in tqdm(range(X_test.shape[0]), ascii=True):
                    
                    y_tr_pr_sum = np.zeros((SX, SY, SZ, n_class))
                    y_tr_pr_cnt = np.zeros((SX, SY, SZ, n_class))
                    
                    for lx in lx_list:
                        for ly in ly_list:
                            for lz in lz_list:
                                
                                    batch_x = X_test[i_c:i_c + 1, lx:lx + LX, ly:ly + LY, lz:lz + LZ, :].copy()
                                    
                                    y_tr_pr_cnt[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += 1
                                    
                                    for i_MC in range(n_MC):
                                        pred_temp = sess.run(predicter,  feed_dict={X: batch_x, p_keep_conv: keep_uncert})
                                        y_tr_pr_sum[lx:lx + LX, ly:ly + LY, lz:lz + LZ,:] += pred_temp[0, :, :, :, :]
                    
                    y_tr_pr_sum/= n_MC
                    
                    y_tr_pr_sum[y_tr_pr_cnt > 0] = y_tr_pr_sum[y_tr_pr_cnt > 0] / y_tr_pr_cnt[y_tr_pr_cnt > 0]
                    
                    y_entr= y_tr_pr_sum*np.log(y_tr_pr_sum+1e-10)
                    y_entr[y_tr_pr_cnt == 0] = 0
                    y_entr= - np.sum( y_entr , axis=3)
                    
                    batch_x = X_test[i_c:i_c + 1, :, :, :, :].copy()
                    batch_y = Y_test[i_c:i_c + 1, :, :, :, :].copy()
                    y_t_c = np.argmax( batch_y[0, :, :, :, :], axis=-1)
                    
                    y_tr_pr_c = np.argmax(y_tr_pr_sum, axis=-1)
                    y_tr_pr_c[y_tr_pr_cnt[:,:,:,0] == 0] = 0
                                        
                    if np.sum(y_tr_pr_c)>0:
                
                        for j_c in range(n_class):
                            dice_c[i_c, j_c] = dk_seg.dice(batch_y[0, :, :, :, j_c] == 1, y_tr_pr_c == j_c)
                        
                        dice_c[i_c, n_class] = hd95(batch_y[0, :, :, :, 1], y_tr_pr_c)
                        dice_c[i_c, n_class+1] = asd(batch_y[0, :, :, :, 1], y_tr_pr_c)
                        dice_c[i_c, n_class+2] = assd(batch_y[0, :, :, :, 1], y_tr_pr_c)
                        
                        error_mask= dk_aux.seg_2_anulus(y_t_c, radius= 2.0)
                        
                        dice_c[i_c, n_class+3] = np.mean(y_entr)
                        temp= y_entr[y_tr_pr_c>0.5]
                        dice_c[i_c, n_class+4] = temp.mean()
                        temp= y_entr[error_mask]
                        dice_c[i_c, n_class+5] = temp.mean()
                        
                        y_tr_pr_soft= y_tr_pr_sum[:,:,:,1]
                        
                        plot_save_path= None
                        ECE, MCE, ECE_curve= dk_aux.estimate_ECE_and_MCE(y_t_c, y_tr_pr_soft, plot_save_path=plot_save_path)
                        dice_c[i_c, n_class+6]= ECE
                        dice_c[i_c, n_class+7]= MCE
                        
                        plot_save_path= None
                        ECE, MCE, ECE_curve= dk_aux.estimate_ECE_and_MCE_masked(y_t_c, y_tr_pr_soft, error_mask, plot_save_path=plot_save_path)
                        dice_c[i_c, n_class+8]= ECE
                        dice_c[i_c, n_class+9]= MCE
                        
                        #dk_aux.save_pred_uncr_thumbs(batch_x[0,:,:,:,0], y_t_c, y_tr_pr_c, y_entr, False, i_c, i_eval, images_dir)
                        
                        #dk_aux.save_uncrt_soft_thumbs(batch_x[0,:,:,:,0], y_t_c, y_tr_pr_c, y_tr_pr_soft, False, i_c, i_eval, images_dir)
                        
                        #Y_pred_2[i_c, :,:,:]= y_tr_pr_soft.copy()
                
                print('test dice with uncertainty estimation  %.3f' % dice_c[:, 0].mean(), ', %.3f' % dice_c[:, 1].mean(), ', %.3f' % dice_c[:, 2].mean())
                
                np.savetxt(results_dir + 'stats_test_' + str(i_eval) + '_uncert.txt', dice_c, fmt='%6.3f', delimiter=',')

                
            
            
            if True: #np.mean(dice_c[:,1])>best_test:
                print('Saving new model checkpoint.')
                best_test = np.mean(dice_c[:,1])
                temp_path = results_dir + 'models/model_saved_' + str(i_eval) + '_' + str(int(round(10000.0 * dice_c[:,1].mean()))) + '.ckpt'
                saver.save(sess, temp_path)
                
            if i_eval == 0:
                dice_old = valid_dice
            else:
                if valid_dice < 0.995 * dice_old:
                    L_Rate = L_Rate * 0.90
                dice_old = valid_dice
                
            print('learning rate and mean test dice:  ', L_Rate, dice_old)

            


                
                
       














































