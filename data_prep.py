#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 10 2020

Code to read the images and segmentions and save them in HDF5 formats for 
later training.

Data sources have been indicated in Table 1 of the paper. 

All data are publicly available for free, with the exception of 
CP- younger fetus and CP- older fetus.  To obtain this data, 
please send email to   Ali.Gholipour@childrens.harvard.edu.

"""




from __future__ import division

import numpy as np
import os
#import tensorflow as tf
# import tensorlayer as tl
from os import listdir
from os.path import isfile, join, isdir
#import scipy.io as sio
#from skimage import io
#import skimage
import SimpleITK as sitk
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
#from tqdm import tqdm
import dk_seg
#import dk_model
#import dk_aux
from PIL import Image
import pandas as pd
#from scipy.stats import beta
#from medpy.metric import hd95, assd, asd



n_channel = 1
n_class = 2

SX, SY, SZ= 135, 189, 155




#######################    SPLEEN    ##########################################

spleen_dir = '/media/nerossd2/segment_everything/spleen/'

desired_spacing= (1.5, 1.5, 1.5)

img_files = [f for f in listdir(spleen_dir+'imagesTr/') if isfile(join(spleen_dir+'imagesTr/', f)) and not '._' in f]
img_files.sort()

n_cases= len(img_files)

normalize_by_mask= True

data_file_name= 'spleen_data.h5'

if os.path.exists(spleen_dir + data_file_name):

    print('Reading data')

    h5f = h5py.File(spleen_dir + data_file_name, 'r')
    X_vol_s = h5f['X_vol'][:]
    Y_vol_s = h5f['Y_vol'][:]
    info_vol_s = h5f['info_vol'][:]
    h5f.close()

    print('Finished reading data')

else:

    X_vol = np.zeros((n_cases, SX, SY, SZ, n_channel), np.float32)
    Y_vol = np.zeros((n_cases, SX, SY, SZ, n_class), np.int8)
    info_vol= np.zeros( (n_cases, 24) )
    i_vol= -1
    
    for img_file in img_files:
        
        i_vol+= 1
        
        vol = sitk.ReadImage(spleen_dir+'imagesTr/' + img_file)
        vol_np = sitk.GetArrayFromImage(vol)
        vol_np = np.transpose(vol_np, [2, 1, 0])
        
        vol_spacing= vol.GetSpacing()
        vol_size   = vol.GetSize()
        
        seg = sitk.ReadImage(spleen_dir+'labelsTr/' + img_file)
        seg_np = sitk.GetArrayFromImage(seg)
        seg_np = np.transpose(seg_np, [2, 1, 0])
        
        assert( np.allclose( vol.GetSpacing(), seg.GetSpacing() ))
        
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        temp = vol_np[vol_np > 0]
        vol_mean= temp.mean()
        vol_std= temp.std()
        
        info_vol[i_vol,0:3]= vol_spacing
        info_vol[i_vol,3:6]= vol_size
        info_vol[i_vol,6:12]= seg_extent
        info_vol[i_vol,12:15]= np.array(vol_spacing) * np.array(vol_size)
        info_vol[i_vol,15:18]= np.array(vol_spacing) * (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        vol= dk_seg.resample3d(vol, vol_spacing, desired_spacing, sitk.sitkBSpline)
        vol_np = sitk.GetArrayFromImage(vol)
        vol_np = np.transpose(vol_np, [2, 1, 0])
        
        seg= dk_seg.resample3d(seg, vol_spacing, desired_spacing, sitk.sitkNearestNeighbor)
        seg_np = sitk.GetArrayFromImage(seg)
        seg_np = np.transpose(seg_np, [2, 1, 0])
        
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        info_vol[i_vol,18:21]= (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        vol_np, seg_np= dk_seg.crop_vol_n_seg(vol_np, seg_np, SX, SY, SZ)
        
        vol_np[vol_np<0]= 0
        vol_np = vol_np / vol_std
        
        X_vol[i_vol, :, :, :, 0] = vol_np.copy()
        Y_vol[i_vol, :, :, :, 1] = seg_np.copy()
        Y_vol[i_vol, :, :, :, 0]= 1- seg_np
        
    h5f = h5py.File(spleen_dir + data_file_name,'w')
    h5f['X_vol']= X_vol
    h5f['Y_vol']= Y_vol
    h5f['info_vol']= info_vol
    h5f.close()




#######################    HEART     ##########################################

heart_dir = '/media/nerossd2/segment_everything/heart/'

img_files = [f for f in listdir(heart_dir+'imagesTr/') if isfile(join(heart_dir+'imagesTr/', f)) and not '._' in f]
img_files.sort()

n_cases= len(img_files)

normalize_by_mask= True

data_file_name= 'heart_data.h5'

if os.path.exists(heart_dir + data_file_name):

    print('Reading data')

    h5f = h5py.File(heart_dir + data_file_name, 'r')
    X_vol_h = h5f['X_vol'][:]
    Y_vol_h = h5f['Y_vol'][:]
    info_vol_h = h5f['info_vol'][:]
    h5f.close()

    print('Finished reading data')

else:

    X_vol = np.zeros((n_cases, SX, SY, SZ, n_channel), np.float32)
    Y_vol = np.zeros((n_cases, SX, SY, SZ, n_class), np.int8)
    info_vol= np.zeros( (n_cases, 24) )
    i_vol= -1
    
    for img_file in img_files:
        
        i_vol+= 1
        
        vol = sitk.ReadImage(heart_dir+'imagesTr/' + img_file)
        vol_np = sitk.GetArrayFromImage(vol)
        vol_np = np.transpose(vol_np, [2, 1, 0])
        
        vol_spacing= vol.GetSpacing()
        vol_size   = vol.GetSize()
        
        seg = sitk.ReadImage(heart_dir+'labelsTr/' + img_file)
        seg_np = sitk.GetArrayFromImage(seg)
        seg_np = np.transpose(seg_np, [2, 1, 0])
        
        assert( np.allclose( vol.GetSpacing(), seg.GetSpacing() ))
        
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        temp = vol_np[vol_np != 0]
        vol_mean= temp.mean()
        vol_std= temp.std()
        
        info_vol[i_vol,0:3]= vol_spacing
        info_vol[i_vol,3:6]= vol_size
        info_vol[i_vol,6:12]= seg_extent
        info_vol[i_vol,12:15]= np.array(vol_spacing) * np.array(vol_size)
        info_vol[i_vol,15:18]= np.array(vol_spacing) * (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        #vol= dk_seg.resample3d(vol, vol_spacing, desired_spacing, sitk.sitkBSpline)
        vol_np = sitk.GetArrayFromImage(vol)
        vol_np = np.transpose(vol_np, [2, 1, 0])
        
        #seg= dk_seg.resample3d(seg, vol_spacing, desired_spacing, sitk.sitkNearestNeighbor)
        seg_np = sitk.GetArrayFromImage(seg)
        seg_np = np.transpose(seg_np, [2, 1, 0])
        
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        info_vol[i_vol,18:21]= (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        vol_np, seg_np= dk_seg.crop_vol_n_seg(vol_np, seg_np, SX, SY, SZ)
        
        vol_np[vol_np<0]= 0
        vol_np = vol_np / vol_std
        
        X_vol[i_vol, :, :, :, 0] = vol_np.copy()
        Y_vol[i_vol, :, :, :, 1] = seg_np.copy()
        Y_vol[i_vol, :, :, :, 0]= 1- seg_np
        
    h5f = h5py.File(heart_dir + data_file_name,'w')
    h5f['X_vol']= X_vol
    h5f['Y_vol']= Y_vol
    h5f['info_vol']= info_vol
    h5f.close()



#######################    PROSTATE     #######################################

prostate_dir = '/media/nerossd2/segment_everything/prostate/'

desired_spacing= (1.25, 1.25, 1.25)

img_files = [f for f in listdir(prostate_dir+'imagesTr/') if isfile(join(prostate_dir+'imagesTr/', f)) and not '._' in f]
img_files.sort()

n_cases= len(img_files)

n_channel= 2
n_class= 2

normalize_by_mask= True

data_file_name= 'prostate_data.h5'

if os.path.exists(prostate_dir + data_file_name):
    
    print('Reading data')
    
    h5f = h5py.File(prostate_dir + data_file_name, 'r')
    X_vol_p = h5f['X_vol'][:]
    Y_vol_p = h5f['Y_vol'][:]
    info_vol_p = h5f['info_vol'][:]
    h5f.close()
    
    print('Finished reading data')
    
else:
    
    X_vol = np.zeros((n_cases, SX, SY, SZ, n_channel), np.float32)
    Y_vol = np.zeros((n_cases, SX, SY, SZ, n_class), np.int8)
    info_vol= np.zeros( (n_cases, 24) )
    i_vol= -1
    
    for img_file in img_files:
        
        i_vol+= 1
        
        vol = sitk.ReadImage(prostate_dir+'imagesTr/' + img_file)
        vol_np = sitk.GetArrayFromImage(vol)
        vol_np = np.transpose(vol_np, [3, 2, 1, 0])
        
        vol_spacing= vol.GetSpacing()[:3]
        vol_size   = vol.GetSize()[:3]
        
        seg = sitk.ReadImage(prostate_dir+'labelsTr/' + img_file)
        seg_np = sitk.GetArrayFromImage(seg)
        seg_np = np.transpose(seg_np, [2, 1, 0])
        
        assert( np.allclose( vol.GetSpacing()[:3], seg.GetSpacing() ))
        
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        info_vol[i_vol,0:3]= vol_spacing
        info_vol[i_vol,3:6]= vol_size
        info_vol[i_vol,6:12]= seg_extent
        info_vol[i_vol,12:15]= np.array(vol_spacing) * np.array(vol_size)
        info_vol[i_vol,15:18]= np.array(vol_spacing) * (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        vol_np_0= vol_np[:,:,:,0].copy()
        vol_np_0 = np.transpose(vol_np_0, [2, 1, 0])
        vol_0= sitk.GetImageFromArray(vol_np_0)
        vol_0.SetSpacing(vol.GetSpacing()[:3])
        vol_0= dk_seg.resample3d(vol_0, vol_spacing, desired_spacing, sitk.sitkBSpline)
        vol_np_0 = sitk.GetArrayFromImage(vol_0)
        vol_np_0 = np.transpose(vol_np_0, [2, 1, 0])
        
        vol_np_1= vol_np[:,:,:,1]
        vol_np_1 = np.transpose(vol_np_1, [2, 1, 0])
        vol_1= sitk.GetImageFromArray(vol_np_1)
        vol_1.SetSpacing(vol.GetSpacing()[:3])
        vol_1= dk_seg.resample3d(vol_1, vol_spacing, desired_spacing, sitk.sitkBSpline)
        vol_np_1 = sitk.GetArrayFromImage(vol_1)
        vol_np_1 = np.transpose(vol_np_1, [2, 1, 0])
        
        seg= dk_seg.resample3d(seg, vol_spacing, desired_spacing, sitk.sitkNearestNeighbor)
        seg_np = sitk.GetArrayFromImage(seg)
        seg_np = np.transpose(seg_np, [2, 1, 0])
        seg_np[seg_np>0]= 1
        
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        info_vol[i_vol,18:21]= (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        seg_np_temp= seg_np.copy()
        vol_np_0, seg_np= dk_seg.crop_vol_n_seg(vol_np_0, seg_np, SX, SY, SZ)
        vol_np_1, _ = dk_seg.crop_vol_n_seg(vol_np_1, seg_np_temp, SX, SY, SZ)
        
        temp = vol_np_0[vol_np_0 != 0]
        vol_std= temp.std()
        vol_np_0 = vol_np_0 / vol_std
        
        temp = vol_np_1[vol_np_1 != 0]
        vol_std= temp.std()
        vol_np_1 = vol_np_1 / vol_std
        
        X_vol[i_vol, :, :, :, 0] = vol_np_0.copy()
        X_vol[i_vol, :, :, :, 1] = vol_np_1.copy()
        Y_vol[i_vol, :, :, :, 1] = seg_np.copy()
        Y_vol[i_vol, :, :, :, 0]= 1- seg_np
        
    h5f = h5py.File(prostate_dir + data_file_name,'w')
    h5f['X_vol']= X_vol
    h5f['Y_vol']= Y_vol
    h5f['info_vol']= info_vol
    h5f.close()




#######################    HIPPOCAMPUS    #####################################

hippocampus_dir = '/media/nerossd2/segment_everything/hippocampus/'

desired_spacing= (0.75, 0.75, 0.75)

img_files = [f for f in listdir(hippocampus_dir+'imagesTr/') if isfile(join(hippocampus_dir+'imagesTr/', f)) and not '._' in f]
img_files.sort()

n_cases= len(img_files)

n_channel= 1
n_class= 3

normalize_by_mask= True

data_file_name= 'hippocampus_data.h5'

if os.path.exists(hippocampus_dir + data_file_name):
    
    print('Reading data')
    
    h5f = h5py.File(hippocampus_dir + data_file_name, 'r')
    X_vol_c = h5f['X_vol'][:]
    Y_vol_c = h5f['Y_vol'][:]
    info_vol_c = h5f['info_vol'][:]
    h5f.close()
    
    print('Finished reading data')
    
else:

    X_vol = np.zeros((n_cases, SX, SY, SZ, n_channel), np.float32)
    Y_vol = np.zeros((n_cases, SX, SY, SZ, n_class), np.int8)
    info_vol= np.zeros( (n_cases, 24) )
    i_vol= -1
    
    for img_file in img_files:
        
        i_vol+= 1
        
        vol = sitk.ReadImage(hippocampus_dir+'imagesTr/' + img_file)
        vol_np = sitk.GetArrayFromImage(vol)
        vol_np = np.transpose(vol_np, [2, 1, 0])
        
        vol_spacing= vol.GetSpacing()
        vol_size   = vol.GetSize()
        
        seg = sitk.ReadImage(hippocampus_dir+'labelsTr/' + img_file)
        seg_np = sitk.GetArrayFromImage(seg)
        seg_np = np.transpose(seg_np, [2, 1, 0])
        
        assert( np.allclose( vol.GetSpacing(), seg.GetSpacing() ))
        
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        temp = vol_np[vol_np > 0]
        vol_mean= temp.mean()
        vol_std= temp.std()
        
        info_vol[i_vol,0:3]= vol_spacing
        info_vol[i_vol,3:6]= vol_size
        info_vol[i_vol,6:12]= seg_extent
        info_vol[i_vol,12:15]= np.array(vol_spacing) * np.array(vol_size)
        info_vol[i_vol,15:18]= np.array(vol_spacing) * (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        vol= dk_seg.resample3d(vol, vol_spacing, desired_spacing, sitk.sitkBSpline)
        vol_np = sitk.GetArrayFromImage(vol)
        vol_np = np.transpose(vol_np, [2, 1, 0])
        
        seg= dk_seg.resample3d(seg, vol_spacing, desired_spacing, sitk.sitkNearestNeighbor)
        seg_np = sitk.GetArrayFromImage(seg)
        seg_np = np.transpose(seg_np, [2, 1, 0])
        
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        info_vol[i_vol,18:21]= (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        vol_np, seg_np= dk_seg.crop_vol_n_seg(vol_np, seg_np, SX, SY, SZ)
        
        vol_np[vol_np<0]= 0
        vol_np = vol_np / vol_std
        
        X_vol[i_vol, :, :, :, 0] = vol_np.copy()
        Y_vol[i_vol, :, :, :, 1] = (seg_np==1).astype(np.int8)
        Y_vol[i_vol, :, :, :, 2] = (seg_np==2).astype(np.int8)
        Y_vol[i_vol, :, :, :, 0]= 1- Y_vol[i_vol, :, :, :, 1] - Y_vol[i_vol, :, :, :, 2]
        
    h5f = h5py.File(hippocampus_dir + data_file_name,'w')
    h5f['X_vol']= X_vol
    h5f['Y_vol']= Y_vol
    h5f['info_vol']= info_vol
    h5f.close()






#######################     pancreas         ##################################

pancreas_dir = '/media/nerossd2/segment_everything/Pancreas/'

desired_spacing= (2.0, 2.0, 2.0)
SX, SY, SZ= 160, 160, 160


img_files = [f for f in listdir(pancreas_dir+'imagesTr/') if isfile(join(pancreas_dir+'imagesTr/', f)) and not '._' in f]
img_files.sort()

n_cases= len(img_files)

n_channel= 1
n_class= 3

normalize_by_mask= True

data_file_name= 'pancreas_data_decath.h5'

if os.path.exists(pancreas_dir + data_file_name):
    
    print('Reading data')
    
    h5f = h5py.File(pancreas_dir + data_file_name, 'r')
    X_vol_r = h5f['X_vol'][:]
    Y_vol_r = h5f['Y_vol'][:]
    info_vol_r = h5f['info_vol'][:]
    h5f.close()
    
    print('Finished reading data')
    
else:
    
    X_vol = np.zeros((n_cases, SX, SY, SZ, n_channel), np.float32)
    Y_vol = np.zeros((n_cases, SX, SY, SZ, n_class), np.int8)
    info_vol= np.zeros( (n_cases, 24) )
    i_vol= -1
    
    for img_file in img_files:
        
        i_vol+= 1
        print(i_vol)
        
        vol = sitk.ReadImage(pancreas_dir+'imagesTr/' + img_file)
        vol_np = sitk.GetArrayFromImage(vol)
        vol_np = np.transpose(vol_np, [2, 1, 0])
        
        vol_spacing= vol.GetSpacing()
        vol_size   = vol.GetSize()
        
        seg = sitk.ReadImage(pancreas_dir+'labelsTr/' + img_file)
        seg_np = sitk.GetArrayFromImage(seg)
        seg_np = np.transpose(seg_np, [2, 1, 0])
        
        assert( np.allclose( vol.GetSpacing()[:3], seg.GetSpacing() ))
        
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        temp = vol_np[vol_np > 0]
        vol_mean= temp.mean()
        vol_std= temp.std()
        
        info_vol[i_vol,0:3]= vol_spacing
        info_vol[i_vol,3:6]= vol_size
        info_vol[i_vol,6:12]= seg_extent
        info_vol[i_vol,12:15]= np.array(vol_spacing) * np.array(vol_size)
        info_vol[i_vol,15:18]= np.array(vol_spacing) * (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        vol= dk_seg.resample3d(vol, vol_spacing, desired_spacing, sitk.sitkBSpline)
        vol_np = sitk.GetArrayFromImage(vol)
        vol_np = np.transpose(vol_np, [2, 1, 0])
        
        seg= dk_seg.resample3d(seg, vol_spacing, desired_spacing, sitk.sitkNearestNeighbor)
        seg_np = sitk.GetArrayFromImage(seg)
        seg_np = np.transpose(seg_np, [2, 1, 0])
        
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        info_vol[i_vol,18:21]= (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        vol_np, seg_np= dk_seg.crop_vol_n_seg_centered(vol_np, seg_np, SX, SY, SZ)
        
        vol_np0= vol_np==0
        vol_np+= 1000
        vol_np = vol_np / 2000
        vol_np[vol_np0]= 0
        
        X_vol[i_vol, :, :, :, 0] = vol_np.copy()
        Y_vol[i_vol, :, :, :, 0] = seg_np==0
        Y_vol[i_vol, :, :, :, 1] = seg_np==1
        Y_vol[i_vol, :, :, :, 2] = seg_np==2
        
    h5f = h5py.File(pancreas_dir + data_file_name,'w')
    h5f['X_vol']= X_vol
    h5f['Y_vol']= Y_vol
    h5f['info_vol']= info_vol
    h5f.close()










#######################    Liver CT       #####################################

Silver_dir = '/media/nerossd2/segment_everything/Silver/'

desired_spacing= (2.5, 2.5, 2.5)

n_cases= 19

n_channel= 1
n_class= 2

normalize_by_mask= True

data_file_name= 'Silver_data.h5'

if os.path.exists(Silver_dir + data_file_name):
    
    print('Reading data')
    
    h5f = h5py.File(Silver_dir + data_file_name, 'r')
    X_vol_sil = h5f['X_vol'][:]
    Y_vol_sil = h5f['Y_vol'][:]
    info_vol_sil = h5f['info_vol'][:]
    h5f.close()
    
    print('Finished reading data')
    
else:

    X_vol = np.zeros((n_cases, SX, SY, SZ, n_channel), np.float32)
    Y_vol = np.zeros((n_cases, SX, SY, SZ, n_class), np.int8)
    info_vol= np.zeros( (n_cases, 24) )
    
    for i_vol in range(n_cases):
        
        if i_vol<9:
            img_file= 'liver-orig00' + str(i_vol+1) + '.mhd'
            seg_file= 'liver-seg00' + str(i_vol+1) + '.mhd'
        else:
            img_file= 'liver-orig0' + str(i_vol+1) + '.mhd'
            seg_file= 'liver-seg0' + str(i_vol+1) + '.mhd'
        
        vol = sitk.ReadImage(Silver_dir + img_file)
        vol_np = sitk.GetArrayFromImage(vol)
        vol_np = np.transpose(vol_np, [2, 1, 0])
        
        vol_spacing= vol.GetSpacing()
        vol_size   = vol.GetSize()
        
        seg = sitk.ReadImage(Silver_dir + seg_file)
        seg_np = sitk.GetArrayFromImage(seg)
        seg_np = np.transpose(seg_np, [2, 1, 0])
        
        assert( np.allclose( vol.GetSpacing(), seg.GetSpacing() ))
        
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        temp = vol_np[vol_np > 0]
        vol_mean= temp.mean()
        vol_std= temp.std()
        
        info_vol[i_vol,0:3]= vol_spacing
        info_vol[i_vol,3:6]= vol_size
        info_vol[i_vol,6:12]= seg_extent
        info_vol[i_vol,12:15]= np.array(vol_spacing) * np.array(vol_size)
        info_vol[i_vol,15:18]= np.array(vol_spacing) * (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        vol= dk_seg.resample3d(vol, vol_spacing, desired_spacing, sitk.sitkBSpline)
        vol_np = sitk.GetArrayFromImage(vol)
        vol_np = np.transpose(vol_np, [1, 2, 0])
        
        seg= dk_seg.resample3d(seg, vol_spacing, desired_spacing, sitk.sitkNearestNeighbor)
        seg_np = sitk.GetArrayFromImage(seg)
        seg_np = np.transpose(seg_np, [1, 2, 0])
        
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        info_vol[i_vol,18:21]= (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        vol_np, seg_np= dk_seg.crop_vol_n_seg_centered(vol_np, seg_np, SX, SY, SZ)
        
        vol_np[vol_np<-200]= 0
        vol_np = vol_np / vol_std
        
        X_vol[i_vol, :, :, :, 0] = vol_np.copy()
        Y_vol[i_vol, :, :, :, 1] = seg_np
        Y_vol[i_vol, :, :, :, 0]= 1- Y_vol[i_vol, :, :, :, 1]
        
    h5f = h5py.File(Silver_dir + data_file_name,'w')
    h5f['X_vol']= X_vol
    h5f['Y_vol']= Y_vol
    h5f['info_vol']= info_vol
    h5f.close()















#######################    Liver-MRI-SPIR    ##################################

CHAOS_dir = '/media/nerossd2/segment_everything/CHAOS/CHAOS_Train_Sets/Train_Sets/MR/'

img_dirs = [d for d in listdir(CHAOS_dir) if isdir(join(CHAOS_dir, d)) ]
img_dirs.sort()

desired_spacing= (2.5, 2.5, 2.5)

n_cases= 20

n_channel= 1
n_class= 2

normalize_by_mask= True

data_file_name= 'CHAOS_data.h5'

if os.path.exists(CHAOS_dir + data_file_name):
    
    print('Reading data')
    
    h5f = h5py.File(CHAOS_dir + data_file_name, 'r')
    X_vol_chaos = h5f['X_vol'][:]
    Y_vol_chaos = h5f['Y_vol'][:]
    info_vol_chaos = h5f['info_vol'][:]
    h5f.close()
    
    print('Finished reading data')
    
else:
    
    X_vol = np.zeros((n_cases, SX, SY, SZ, n_channel), np.float32)
    Y_vol = np.zeros((n_cases, SX, SY, SZ, n_class), np.int8)
    info_vol= np.zeros( (n_cases, 24) )
    i_vol= -1
    
    for img_dir in img_dirs:
        
        i_vol+= 1
        
        case_dir= CHAOS_dir + img_dir + '/T2SPIR/DICOM_anon/'
        seg_dir=  CHAOS_dir + img_dir + '/T2SPIR/Ground/'
        
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames( case_dir )
        reader.SetFileNames(dicom_names)
        vol = reader.Execute()
        
        vol_np = sitk.GetArrayFromImage(vol)
        vol_np = np.transpose(vol_np, [2, 1, 0])
        
        vol_spacing= vol.GetSpacing()
        vol_size   = vol.GetSize()
        
        seg_np= np.zeros( vol_np.shape )
        seg_files = [f for f in listdir(seg_dir) if isfile(join(seg_dir, f)) ]
        assert( seg_np.shape[-1]==len(seg_files))
        seg_files.sort()
        for i_seg in range(seg_np.shape[-1]):
            seg_slice = Image.open( seg_dir +  seg_files[i_seg])
            seg_slice= np.array(seg_slice).T
            seg_slice= seg_slice==63
            seg_np[:,:,i_seg]= seg_slice.astype(np.int)
            
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        seg_np = np.transpose(seg_np, [2, 1, 0])
        seg = sitk.GetImageFromArray(seg_np)
        seg.SetDirection(vol.GetDirection())
        seg.SetOrigin(vol.GetOrigin())
        seg.SetSpacing(vol.GetSpacing())
        
        temp = vol_np[vol_np > 0]
        vol_mean= temp.mean()
        vol_std= temp.std()
        
        info_vol[i_vol,0:3]= vol_spacing
        info_vol[i_vol,3:6]= vol_size
        info_vol[i_vol,6:12]= seg_extent
        info_vol[i_vol,12:15]= np.array(vol_spacing) * np.array(vol_size)
        info_vol[i_vol,15:18]= np.array(vol_spacing) * (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        vol= dk_seg.resample3d(vol, vol_spacing, desired_spacing, sitk.sitkBSpline)
        vol_np = sitk.GetArrayFromImage(vol)
        vol_np = np.transpose(vol_np, [1, 2, 0])
        
        seg= dk_seg.resample3d(seg, vol_spacing, desired_spacing, sitk.sitkNearestNeighbor)
        seg_np = sitk.GetArrayFromImage(seg)
        seg_np = np.transpose(seg_np, [1, 2, 0])
        
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        info_vol[i_vol,18:21]= (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        vol_np, seg_np= dk_seg.crop_vol_n_seg_centered(vol_np, seg_np, SX, SY, SZ)
        
        vol_np[vol_np<-200]= 0
        vol_np = vol_np / vol_std
        
        X_vol[i_vol, :, :, :, 0] = vol_np.copy()
        Y_vol[i_vol, :, :, :, 1] = seg_np
        Y_vol[i_vol, :, :, :, 0]= 1- Y_vol[i_vol, :, :, :, 1]
        
    h5f = h5py.File(CHAOS_dir + data_file_name,'w')
    h5f['X_vol']= X_vol
    h5f['Y_vol']= Y_vol
    h5f['info_vol']= info_vol
    h5f.close()












##########    Liver-MRI-DUAL-in and Liver-MRI-DUAL-out   ######################

CHAOS_dir = '/media/nerossd2/segment_everything/CHAOS/CHAOS_Train_Sets/Train_Sets/MR/'

img_dirs = [d for d in listdir(CHAOS_dir) if isdir(join(CHAOS_dir, d)) ]
img_dirs.sort()

desired_spacing= (2.5, 2.5, 2.5)

n_cases= 20

n_channel= 1
n_class= 2

normalize_by_mask= True

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
    
else:
    
    X_vol_in  = np.zeros((n_cases, SX, SY, SZ, n_channel), np.float32)
    X_vol_out = np.zeros((n_cases, SX, SY, SZ, n_channel), np.float32)
    Y_vol     = np.zeros((n_cases, SX, SY, SZ, n_class), np.int8)
    info_vol= np.zeros( (n_cases, 24) )
    i_vol= -1
    
    for img_dir in img_dirs:
        
        i_vol+= 1
        
        case_dir_in=  CHAOS_dir + img_dir + '/T1DUAL/DICOM_anon/InPhase/'
        case_dir_out= CHAOS_dir + img_dir + '/T1DUAL/DICOM_anon/OutPhase/'
        seg_dir=      CHAOS_dir + img_dir + '/T1DUAL/Ground/'
        
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames( case_dir_in )
        reader.SetFileNames(dicom_names)
        vol_in = reader.Execute()
        
        vol_np_in = sitk.GetArrayFromImage(vol_in)
        vol_np_in = np.transpose(vol_np_in, [2, 1, 0])
        
        vol_spacing= vol_in.GetSpacing()
        vol_size   = vol_in.GetSize()
        
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames( case_dir_out )
        reader.SetFileNames(dicom_names)
        vol_out = reader.Execute()
        
        vol_np_out = sitk.GetArrayFromImage(vol_out)
        vol_np_out = np.transpose(vol_np_out, [2, 1, 0])
        
        assert( np.allclose( vol_np_out.shape, vol_np_out.shape ))
        
        seg_np= np.zeros( vol_np_in.shape )
        seg_files = [f for f in listdir(seg_dir) if isfile(join(seg_dir, f)) ]
        assert( seg_np.shape[-1]==len(seg_files))
        seg_files.sort()
        for i_seg in range(seg_np.shape[-1]):
            seg_slice = Image.open( seg_dir +  seg_files[i_seg])
            seg_slice= np.array(seg_slice).T
            seg_slice= seg_slice==63
            seg_np[:,:,i_seg]= seg_slice.astype(np.int)
            
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        seg_np = np.transpose(seg_np, [2, 1, 0])
        seg = sitk.GetImageFromArray(seg_np)
        seg.SetDirection(vol_in.GetDirection())
        seg.SetOrigin(vol_in.GetOrigin())
        seg.SetSpacing(vol_in.GetSpacing())
        
        info_vol[i_vol,0:3]= vol_spacing
        info_vol[i_vol,3:6]= vol_size
        info_vol[i_vol,6:12]= seg_extent
        info_vol[i_vol,12:15]= np.array(vol_spacing) * np.array(vol_size)
        info_vol[i_vol,15:18]= np.array(vol_spacing) * (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        vol_in= dk_seg.resample3d(vol_in, vol_spacing, desired_spacing, sitk.sitkBSpline)
        vol_np_in = sitk.GetArrayFromImage(vol_in)
        vol_np_in = np.transpose(vol_np_in, [1, 2, 0])
        
        vol_out= dk_seg.resample3d(vol_out, vol_spacing, desired_spacing, sitk.sitkBSpline)
        vol_np_out = sitk.GetArrayFromImage(vol_out)
        vol_np_out = np.transpose(vol_np_out, [1, 2, 0])
        
        seg= dk_seg.resample3d(seg, vol_spacing, desired_spacing, sitk.sitkNearestNeighbor)
        seg_np = sitk.GetArrayFromImage(seg)
        seg_np = np.transpose(seg_np, [1, 2, 0])
        
        z= np.where(seg_np>0)
        seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )
        
        info_vol[i_vol,18:21]= (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
        
        seg_np_copy= seg_np.copy()
        vol_np_in,  seg_np= dk_seg.crop_vol_n_seg_centered(vol_np_in, seg_np, SX, SY, SZ)
        vol_np_out, _ = dk_seg.crop_vol_n_seg_centered(vol_np_out, seg_np_copy, SX, SY, SZ)
        
        temp = vol_np[vol_np_in > 0]
        vol_std= temp.std()
        vol_np_in = vol_np_in / vol_std
        
        temp = vol_np[vol_np_out > 0]
        vol_std= temp.std()
        vol_np_out = vol_np_out / vol_std
        
        X_vol_in[i_vol, :, :, :, 0]  = vol_np_in.copy()
        X_vol_out[i_vol, :, :, :, 0] = vol_np_out.copy()
        Y_vol[i_vol, :, :, :, 1] = seg_np
        Y_vol[i_vol, :, :, :, 0]= 1- Y_vol[i_vol, :, :, :, 1]
        
    h5f = h5py.File(CHAOS_dir + data_file_name,'w')
    h5f['X_vol_in']=  X_vol_in
    h5f['X_vol_out']= X_vol_out
    h5f['Y_vol_in']=  Y_vol
    h5f['Y_vol_out']= Y_vol
    h5f['info_vol']= info_vol
    h5f.close()













#####    DHCP

DHCP_dir = '/media/nerossd2/segment_everything/DHCP/'
dhcp_dir= '/media/davood/bd794557-342d-4d50-8b7f-f6c209248f89/davood/DWI_data/dHCP/data/'
#res_dir=  '/media/davood/bd794557-342d-4d50-8b7f-f6c209248f89/davood/DWI_data/dHCP/results/'

desired_spacing= (0.8, 0.8, 0.8)

n_cases= 400

n_channel= 1
n_class= 2

normalize_by_mask= True

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

else:

    subj_info= pd.read_csv( dhcp_dir + 'participants.tsv', delimiter= '\t')

    X_vol = np.zeros((n_cases, SX, SY, SZ, n_channel), np.float32)
    Y_vol = np.zeros((n_cases, SX, SY, SZ, n_class), np.int8)
    info_vol= np.zeros( (n_cases, 24) )
    subj_tag= list()
    i_vol= -1

    for subj in subj_info['participant_id']:

        anat_dir= dhcp_dir + 'dhcp_anat_pipeline/sub-' + subj

        anat_sess = [d for d in listdir(anat_dir) if isdir(join(anat_dir, d))]

        sess_tsv= anat_dir + '/sub-' + subj + '_sessions.tsv'
        sess_info= pd.read_csv( sess_tsv , delimiter= '\t')
        ages= np.array(sess_info['scan_age'])

        for j in range(len(sess_info)):

            i_vol+= 1

            age_c= sess_info.loc[j, 'scan_age']

            print('\n'*0, i_vol, '  Processing subject: ', subj, ',   session:', sess_info.loc[j, 'session_id'], ',    age:', age_c)

            subject= 'sub-' + subj
            session= 'ses-'+ str(sess_info.loc[j, 'session_id'])

            ant_dir = anat_dir + '/' + session + '/' + 'anat/'

            file_name= subject + '_' + session + '_desc-restore_T2w.nii.gz'
            vol= sitk.ReadImage( ant_dir + file_name )
            vol_np= sitk.GetArrayFromImage(vol)
            vol_np= np.transpose( vol_np, [2, 1 ,0] )

            vol_spacing= vol.GetSpacing()
            vol_size   = vol.GetSize()

            file_name= subject + '_' + session + '_desc-drawem9_space-T2w_dseg.nii.gz'
            seg= sitk.ReadImage( ant_dir + file_name )
            seg_np= sitk.GetArrayFromImage(seg)
            seg_np= (seg_np==2).astype(np.int)
            seg_np= np.transpose( seg_np, [2, 1, 0] )

            assert( np.allclose( vol.GetSpacing(), seg.GetSpacing() ))

            z= np.where(seg_np>0)
            seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )

            temp = vol_np[vol_np > 0]
            vol_std= temp.std()

            subj_tag.append(subj)
            info_vol[i_vol,-1]= age_c
            info_vol[i_vol,0:3]= vol_spacing
            info_vol[i_vol,3:6]= vol_size
            info_vol[i_vol,6:12]= seg_extent
            info_vol[i_vol,12:15]= np.array(vol_spacing) * np.array(vol_size)
            info_vol[i_vol,15:18]= np.array(vol_spacing) * (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))

            vol= dk_seg.resample3d(vol, vol_spacing, desired_spacing, sitk.sitkBSpline)
            vol_np = sitk.GetArrayFromImage(vol)
            vol_np = np.transpose(vol_np, [2, 1, 0])

            seg= dk_seg.resample3d(seg, vol_spacing, desired_spacing, sitk.sitkNearestNeighbor)
            seg_np = sitk.GetArrayFromImage(seg)
            seg_np = np.transpose(seg_np, [2, 1, 0])

            vol_np= vol_np*(seg_np>0)
            seg_np= (seg_np==2).astype(np.int8)

            vol_np= vol_np[:,::-1,:]
            seg_np= seg_np[:,::-1,:]

            z= np.where(seg_np>0)
            seg_extent= ( z[0].min() , z[0].max() , z[1].min() , z[1].max() , z[2].min() , z[2].max() )

            info_vol[i_vol,18:21]= (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))

            vol_np, seg_np= dk_seg.crop_vol_n_seg_centered(vol_np, seg_np, SX, SY, SZ)

            vol_np = vol_np / vol_std

            X_vol[i_vol, :, :, :, 0] = vol_np.copy()
            Y_vol[i_vol, :, :, :, 1] = seg_np
            Y_vol[i_vol, :, :, :, 0]= 1- Y_vol[i_vol, :, :, :, 1]

    h5f = h5py.File(DHCP_dir + data_file_name,'w')
    h5f['X_vol']= X_vol
    h5f['Y_vol']= Y_vol
    h5f['info_vol']= info_vol
    h5f.close()

    with open(DHCP_dir + 'subject_tag.txt' , 'wb') as f:
        pickle.dump(subj_tag, f)













