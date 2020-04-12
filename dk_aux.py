# -*- coding: utf-8 -*-
"""

Helpers for plotting, saving, etc.

@author: davoo
"""


import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import os
from scipy.spatial.distance import directed_hausdorff
import dk_seg
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics

def save_data_thumbs(x_c, y_t_c, y_a_c, image_index, thumbs_dir, n_class=2):

    n_rows, n_cols = 3, 3

    z = np.where(y_t_c > 0)
    SX, SY, SZ = ( z[0].min()+  z[0].max() ), ( z[1].min() + z[1].max() ) , ( z[2].min() + z[2].max())

    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)

    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray')
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 7)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(y_t_c[:, :, SZ // 2], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 5)
    plt.imshow(y_t_c[:, SY // 2, :], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 8)
    plt.imshow(y_t_c[SX // 2, :, :], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(y_a_c[:, :, SZ // 2], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 6)
    plt.imshow(y_a_c[:, SY // 2, :], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 9)
    plt.imshow(y_a_c[SX // 2, :, :], vmin=0, vmax= n_class-1)

    fig.savefig(thumbs_dir + 'Data_' + str(image_index) + '.png')

    plt.close(fig)


def save_pred_thumbs(x_c, y_t_c, y_a_c, training_flag, image_index, iteration_count, prediction_dir, perc_low= 5, perc_hi= 95, n_class=2):
    
    n_rows, n_cols = 3, 3
    
    SX, SY, SZ = x_c.shape
    
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    '''plt.subplot(n_rows, n_cols, 1)
    vmin, vmax= np.percentile(x_c[:, :, SZ // 2], perc_low) , np.percentile(x_c[:, :, SZ // 2], perc_hi)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 4)
    vmin, vmax= np.percentile(x_c[:, SY // 2, :], perc_low) , np.percentile(x_c[:, SY // 2, :], perc_hi)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 7)
    vmin, vmax= np.percentile(x_c[SX // 2, :, :], perc_low) , np.percentile(x_c[SX // 2, : , :], perc_hi)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray', vmin= vmin, vmax= vmax)'''
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray')
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 7)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(y_t_c[:, :, SZ // 2], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 5)
    plt.imshow(y_t_c[:, SY // 2, :], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 8)
    plt.imshow(y_t_c[SX // 2, :, :], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(y_a_c[:, :, SZ // 2], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 6)
    plt.imshow(y_a_c[:, SY // 2, :], vmin=0, vmax= n_class-1)
    plt.subplot(n_rows, n_cols, 9)
    plt.imshow(y_a_c[SX // 2, :, :], vmin=0, vmax= n_class-1)
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
    plt.close(fig)


def save_pred_thumbs_seg_centr(x_c, y_t_c, y_p_c, training_flag, image_index, iteration_count, prediction_dir):
    
    z = np.where(y_t_c > 0)
    
    if len(z[0])<10:
        
        print('Segmentation mask smaller than 10; image index: ', image_index)
        
    else:
        
        x_min, x_max, y_min, y_max, z_min, z_max = z[0].min(), z[0].max(), z[1].min(), z[1].max(), z[2].min(), z[2].max()
        
        SX = (x_min + x_max)
        SY = (y_min + y_max)
        SZ = (z_min + z_max)
        
        n_rows, n_cols = 3, 3
        
        fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
        
        plt.subplot(n_rows, n_cols, 1)
        plt.imshow(x_c[:, :, SZ // 2], cmap='gray')
        plt.subplot(n_rows, n_cols, 2)
        plt.imshow(x_c[:, SY // 2, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 3)
        plt.imshow(x_c[SX // 2, :, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 4)
        plt.imshow(y_t_c[:, :, SZ // 2])
        plt.subplot(n_rows, n_cols, 5)
        plt.imshow(y_t_c[:, SY // 2, :])
        plt.subplot(n_rows, n_cols, 6)
        plt.imshow(y_t_c[SX // 2, :, :])
        plt.subplot(n_rows, n_cols, 7)
        plt.imshow(y_p_c[:, :, SZ // 2])
        plt.subplot(n_rows, n_cols, 8)
        plt.imshow(y_p_c[:, SY // 2, :])
        plt.subplot(n_rows, n_cols, 9)
        plt.imshow(y_p_c[SX // 2, :, :])
        
        if training_flag:
            fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
        else:
            fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
            
        plt.close(fig)



#
# def save_pred_thumbs(x_c, y_t_c, y_a_c, y_p_c, training_flag, image_index, iteration_count, prediction_dir):
#
#     n_rows, n_cols = 3, 4
#
#     SX, SY, SZ= x_c.shape
#
#     fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
#
#     plt.subplot(n_rows, n_cols, 1)
#     plt.imshow(x_c[:, :, SZ // 2], cmap='gray')
#     plt.subplot(n_rows, n_cols, 5)
#     plt.imshow(x_c[:, SY // 2, :], cmap='gray')
#     plt.subplot(n_rows, n_cols, 9)
#     plt.imshow(x_c[SX // 2, :, :], cmap='gray')
#     plt.subplot(n_rows, n_cols, 2)
#     plt.imshow(y_t_c[:, :, SZ // 2])
#     plt.subplot(n_rows, n_cols, 6)
#     plt.imshow(y_t_c[:, SY // 2, :])
#     plt.subplot(n_rows, n_cols, 10)
#     plt.imshow(y_t_c[SX // 2, :, :])
#     plt.subplot(n_rows, n_cols, 3)
#     plt.imshow(y_a_c[:, :, SZ // 2])
#     plt.subplot(n_rows, n_cols, 7)
#     plt.imshow(y_a_c[:, SY // 2, :])
#     plt.subplot(n_rows, n_cols, 11)
#     plt.imshow(y_a_c[SX // 2, :, :])
#     plt.subplot(n_rows, n_cols, 4)
#     plt.imshow(y_p_c[:, :, SZ // 2])
#     plt.subplot(n_rows, n_cols, 8)
#     plt.imshow(y_p_c[:, SY // 2, :])
#     plt.subplot(n_rows, n_cols, 12)
#     plt.imshow(y_p_c[SX // 2, :, :])
#     if training_flag:
#         fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
#     else:
#         fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
#     plt.close(fig)




def save_cm(CM, training_flag, iteration_count, label_tags, prediction_dir):
    
    n_rows, n_cols = 1,1
    
    fig, ax = plt.subplots(figsize=(10, 10), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    ax.get_xaxis().set_visible(False)
    plt.imshow(CM)
    ax.set_xticklabels( ['']+label_tags )
    ax.set_yticklabels( ['']+label_tags )
    
    if training_flag:
        fig.savefig(prediction_dir + 'CM_train_' + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'CM_test_' + '_' + str(iteration_count) + '.png')
    plt.close(fig)



def predict_with_shifts(sess, X, softmax_linear, p_keep_conv, batch_x, n_class, MX, MY, MZ, xs_v, ys_v, zs_v):
    
    _, a_x, b_x, c_x, n_channel = batch_x.shape
    y_prob= np.zeros((a_x + MX, b_x + MY, c_x + MZ, n_class), np.float32)
    i_prob= 0
    
    for xs in xs_v:
        for ys in ys_v:
            for zs in zs_v:
                
                x = batch_x[0, :, :, :, :].copy()
                
                batch_xx= np.zeros(batch_x.shape, dtype=np.float32)
                
                xx = np.zeros((a_x + MX, b_x + MY, c_x + MZ, n_channel), np.float32)
                xx[MX // 2:MX // 2 + a_x, MY // 2:MY // 2 + b_x, MZ // 2:MZ // 2 + c_x,:] = x.copy()
                
                batch_xx[0, :, :, :, :] = xx[xs:xs+a_x, ys:ys+b_x, zs:zs+c_x, :].copy()
                
                y_prob_c= sess.run(softmax_linear, feed_dict={X: batch_xx, p_keep_conv: 1.0})
                
                y_prob[xs:xs+a_x, ys:ys+b_x, zs:zs+c_x, :]+= y_prob_c[0,:,:,:]
                
                i_prob+= 1
                
                x = xx = 0
                
    y_prob= y_prob[MX // 2:MX // 2 + a_x, MY // 2:MY // 2 + b_x, MZ // 2:MZ // 2 + c_x,:]
    y_prob/= i_prob
    
    return y_prob


def my_softmax(x):
    x_max = np.max(x, axis=-1)
    x_max = x_max[:,:,:, np.newaxis]
    x= np.exp(x - x_max)
    x_sum= np.sum(x, axis=-1)
    x_sum = x_sum[:,:,:, np.newaxis]
    return x / x_sum




def save_pred_uncr_thumbs(x_c, y_t_c, y_a_c, y_unc, training_flag, image_index, iteration_count, prediction_dir, perc_low= 5, perc_hi= 95):
    
    n_rows, n_cols = 3, 4
    
    SX, SY, SZ = x_c.shape
    
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    vmin, vmax= np.percentile(x_c[:, :, SZ // 2], perc_low) , np.percentile(x_c[:, :, SZ // 2], perc_hi)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 5)
    vmin, vmax= np.percentile(x_c[:, SY // 2, :], perc_low) , np.percentile(x_c[:, SY // 2, :], perc_hi)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 9)
    vmin, vmax= np.percentile(x_c[SX // 2, :, :], perc_low) , np.percentile(x_c[SX // 2, : , :], perc_hi)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(y_t_c[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 6)
    plt.imshow(y_t_c[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 10)
    plt.imshow(y_t_c[SX // 2, :, :])
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(y_a_c[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 7)
    plt.imshow(y_a_c[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 11)
    plt.imshow(y_a_c[SX // 2, :, :])
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(y_unc[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 8)
    plt.imshow(y_unc[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 12)
    plt.imshow(y_unc[SX // 2, :, :])
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '_uncr.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '_uncr.png')
    plt.close(fig)


def save_pred_soft_thumbs(x_c, y_t_c, y_a_c, y_soft, training_flag, image_index, iteration_count, prediction_dir, perc_low= 5, perc_hi= 95):
    
    n_rows, n_cols = 3, 4
    
    SX, SY, SZ = x_c.shape
    
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    vmin, vmax= np.percentile(x_c[:, :, SZ // 2], perc_low) , np.percentile(x_c[:, :, SZ // 2], perc_hi)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 5)
    vmin, vmax= np.percentile(x_c[:, SY // 2, :], perc_low) , np.percentile(x_c[:, SY // 2, :], perc_hi)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 9)
    vmin, vmax= np.percentile(x_c[SX // 2, :, :], perc_low) , np.percentile(x_c[SX // 2, : , :], perc_hi)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(y_t_c[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 6)
    plt.imshow(y_t_c[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 10)
    plt.imshow(y_t_c[SX // 2, :, :])
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(y_a_c[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 7)
    plt.imshow(y_a_c[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 11)
    plt.imshow(y_a_c[SX // 2, :, :])
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(y_soft[:, :, SZ // 2], vmin=0, vmax=1)
    plt.subplot(n_rows, n_cols, 8)
    plt.imshow(y_soft[:, SY // 2, :], vmin=0, vmax=1)
    plt.subplot(n_rows, n_cols, 12)
    plt.imshow(y_soft[SX // 2, :, :], vmin=0, vmax=1)
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '_soft.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '_soft.png')
    plt.close(fig)




def save_uncrt_soft_thumbs(x_c, y_t_c, y_a_c, y_soft, training_flag, image_index, iteration_count, prediction_dir, perc_low= 5, perc_hi= 95):
    
    n_rows, n_cols = 3, 4
    
    SX, SY, SZ = x_c.shape
    
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    plt.subplot(n_rows, n_cols, 1)
    vmin, vmax= np.percentile(x_c[:, :, SZ // 2], perc_low) , np.percentile(x_c[:, :, SZ // 2], perc_hi)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 5)
    vmin, vmax= np.percentile(x_c[:, SY // 2, :], perc_low) , np.percentile(x_c[:, SY // 2, :], perc_hi)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 9)
    vmin, vmax= np.percentile(x_c[SX // 2, :, :], perc_low) , np.percentile(x_c[SX // 2, : , :], perc_hi)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray', vmin= vmin, vmax= vmax)
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(y_t_c[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 6)
    plt.imshow(y_t_c[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 10)
    plt.imshow(y_t_c[SX // 2, :, :])
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(y_a_c[:, :, SZ // 2])
    plt.subplot(n_rows, n_cols, 7)
    plt.imshow(y_a_c[:, SY // 2, :])
    plt.subplot(n_rows, n_cols, 11)
    plt.imshow(y_a_c[SX // 2, :, :])
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(y_soft[:, :, SZ // 2], vmin=0, vmax=1)
    plt.subplot(n_rows, n_cols, 8)
    plt.imshow(y_soft[:, SY // 2, :], vmin=0, vmax=1)
    plt.subplot(n_rows, n_cols, 12)
    plt.imshow(y_soft[SX // 2, :, :], vmin=0, vmax=1)
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '_uncrt_soft.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '_uncrt_soft.png')
    plt.close(fig)
    
    
    



def save_pred_mhds(x_c, y_t_c, y_p_c, training_flag, image_index, iteration_count, prediction_dir):
    
    if x_c is not None:
        x= np.transpose(x_c, [2,1,0])
        x= sitk.GetImageFromArray(x)
        if training_flag:
            sitk.WriteImage(x, prediction_dir +  'X_train_' + str(image_index) + '_' + str(iteration_count) + '.mhd')
        else:
            sitk.WriteImage(x, prediction_dir +  'X_test_'  + str(image_index) + '_' + str(iteration_count) + '.mhd')
    
    if y_t_c is not None:
        x= np.transpose(y_t_c, [2,1,0])
        x= sitk.GetImageFromArray(x)
        if training_flag:
            sitk.WriteImage(x, prediction_dir +  'X_train_gold' + str(image_index) + '_' + str(iteration_count) + '.mhd')
        else:
            sitk.WriteImage(x, prediction_dir +  'X_test_gold'  + str(image_index) + '_' + str(iteration_count) + '.mhd')
    
    x= np.transpose(y_p_c, [2,1,0])
    x= sitk.GetImageFromArray(x)
    if training_flag:
        sitk.WriteImage(x, prediction_dir +  'X_train_pred' + str(image_index) + '_' + str(iteration_count) + '.mhd')
    else:
        sitk.WriteImage(x, prediction_dir +  'X_test_pred'  + str(image_index) + '_' + str(iteration_count) + '.mhd')



def save_pred_uncr_mhds(x_c, y_t_c, y_p_c, training_flag, image_index, iteration_count, prediction_dir):
    
    x= np.transpose(x_c, [2,1,0])
    x= sitk.GetImageFromArray(x)
    if training_flag:
        sitk.WriteImage(x, prediction_dir +  'X_train_u_f_' + str(image_index) + '_' + str(iteration_count) + '.mhd')
    else:
        sitk.WriteImage(x, prediction_dir +  'X_test_u_f_'  + str(image_index) + '_' + str(iteration_count) + '.mhd')
    
    x= np.transpose(y_t_c, [2,1,0])
    x= sitk.GetImageFromArray(x)
    if training_flag:
        sitk.WriteImage(x, prediction_dir +  'X_train_u_s_' + str(image_index) + '_' + str(iteration_count) + '.mhd')
    else:
        sitk.WriteImage(x, prediction_dir +  'X_test_u_s_'  + str(image_index) + '_' + str(iteration_count) + '.mhd')
    
    x= np.transpose(y_p_c, [2,1,0])
    x= sitk.GetImageFromArray(x)
    if training_flag:
        sitk.WriteImage(x, prediction_dir +  'X_train_u_t_' + str(image_index) + '_' + str(iteration_count) + '.mhd')
    else:
        sitk.WriteImage(x, prediction_dir +  'X_test_u_t_'  + str(image_index) + '_' + str(iteration_count) + '.mhd')




def divide_patint_wise(patient_code, n_fold, i_fold):
    
    patient_code_unq = np.unique(patient_code)
    patient_code_unq.sort()
    
    np.random.seed(0)
    np.random.shuffle(patient_code_unq)
    
    n_patients = len(patient_code_unq)
    
    patient_test = patient_code_unq[i_fold * n_patients // n_fold:(i_fold + 1) * n_patients // n_fold]
    '''patient_train= np.concatenate( ( patient_code_unq[0:i_fold*n_patients//n_fold] , 
                                     patient_code_unq[(i_fold+1)*n_patients//n_fold:] ) )'''
    
    p_test = np.zeros(len(patient_code), np.int)
    p_train = np.zeros(len(patient_code), np.int)
    n_test = n_train = -1
    
    for i in range(len(patient_code)):
        if patient_code[i] in patient_test:
            n_test += 1
            p_test[n_test] = i
        else:
            n_train += 1
            p_train[n_train] = i
            
    return p_test[:n_test + 1], p_train[:n_train + 1]






def divide_patint_wise_with_gold_dwi(patient_code, patient_code_gold, n_fold, i_fold, random=True, train_on_noisy=True):
    
    patient_code_unq = np.unique(patient_code)
    patient_code_unq.sort()
    
    if random:
        np.random.seed(0)
        np.random.shuffle(patient_code_unq)
    
    n_patients = len(patient_code_unq)
    
    patient_test = patient_code_unq[i_fold * n_patients // n_fold:(i_fold + 1) * n_patients // n_fold]
    '''patient_train= np.concatenate( ( patient_code_unq[0:i_fold*n_patients//n_fold] , 
                                     patient_code_unq[(i_fold+1)*n_patients//n_fold:] ) )'''
    
    p_test = np.zeros(len(patient_code), np.int)
    p_train = np.zeros(len(patient_code), np.int)
    n_test = n_train = -1
    
    for i in range(len(patient_code_gold)):
        if patient_code_gold[i] in patient_test:
            n_test += 1
            p_test[n_test] = i
            
    if train_on_noisy:
        for i in range(len(patient_code)):
            if not patient_code[i] in patient_test:
                n_train += 1
                p_train[n_train] = i
    else:
        for i in range(len(patient_code_gold)):
            if not patient_code_gold[i] in patient_test:
                n_train += 1
                p_train[n_train] = i
            
    return p_test[:n_test + 1], p_train[:n_train + 1]




def divide_patint_wise_with_gold(patient_code, patient_code_gold, n_fold, i_fold, random=True):
    
    patient_code_unq = np.unique(patient_code)
    patient_code_unq.sort()
    
    if random:
        np.random.seed(0)
        np.random.shuffle(patient_code_unq)
    
    n_patients = len(patient_code_unq)
    
    patient_test = patient_code_unq[i_fold * n_patients // n_fold:(i_fold + 1) * n_patients // n_fold]
    '''patient_train= np.concatenate( ( patient_code_unq[0:i_fold*n_patients//n_fold] , 
                                     patient_code_unq[(i_fold+1)*n_patients//n_fold:] ) )'''
    
    p_test_clean  = np.zeros(len(patient_code), np.int)
    p_train_clean = np.zeros(len(patient_code), np.int)
    p_test_noisy  = np.zeros(len(patient_code), np.int)
    p_train_noisy = np.zeros(len(patient_code), np.int)
    n_test_clean = n_train_clean = n_test_noisy = n_train_noisy = -1
    
    for i in range(len(patient_code_gold)):
        if patient_code_gold[i] in patient_test:
            n_test_clean += 1
            p_test_clean[n_test_clean] = i
        else:
            n_train_clean += 1
            p_train_clean[n_train_clean] = i
    
    for i in range(len(patient_code)):
        if patient_code[i] in patient_test:
            n_test_noisy += 1
            p_test_noisy[n_test_noisy] = i
        else:
            n_train_noisy += 1
            p_train_noisy[n_train_noisy] = i
    
    return p_test_noisy[:n_test_noisy + 1], p_train_noisy[:n_train_noisy + 1], p_test_clean[:n_test_clean + 1], p_train_clean[:n_train_clean + 1]







def pruning_error(y_true, y_pred, y_uncert=None, uncert_prcnt=None, mode='Dice'):

    if mode == 'Dice':

        dice_num = 2 * np.sum((y_true == 1) * (y_pred == 1)) + 0
        dice_den = np.sum(y_true == 1) + np.sum(y_pred == 1) + 1
        err = - dice_num / dice_den

    elif mode == 'Hausdorff':

        y_true_b = dk_seg.seg_2_boundary_3d(y_true)
        y_pred_b = dk_seg.seg_2_boundary_3d(y_pred)

        z = np.nonzero(y_true_b)
        zx, zy, zz = z[0], z[1], z[2]
        contour_true = np.vstack([zx, zy, zz]).T
        z = np.nonzero(y_pred_b)
        zx, zy, zz = z[0], z[1], z[2]
        contour_pred = np.vstack([zx, zy, zz]).T

        err = max(directed_hausdorff(contour_true, contour_pred)[0],
                  directed_hausdorff(contour_pred, contour_true)[0])

    elif mode == 'Uncertainry':

        y_uncert = y_uncert[y_true == 1]
        err = np.percentile(y_uncert, uncert_prcnt)

    else:

        print('Unrecognized error mode; returning zero.')
        err = 0

    return err



def samples_2_keep(err_tr, prune_perct=95):

    perc = np.percentile(err_tr, prune_perct)

    ind = np.where(err_tr < perc)

    return ind






def gaussian_w_3d(sx, sy, sz, sigx, sigy, sigz):
    
    wx = np.exp(- np.abs(np.linspace(-sx / 2 + 1 / 2, sx / 2 - 1 / 2, sx)) ** 2 / sigx ** 2)
    wy = np.exp(- np.abs(np.linspace(-sy / 2 + 1 / 2, sy / 2 - 1 / 2, sy)) ** 2 / sigy ** 2)
    wz = np.exp(- np.abs(np.linspace(-sz / 2 + 1 / 2, sz / 2 - 1 / 2, sz)) ** 2 / sigz ** 2)
    
    wxy= np.matmul( wx[:,np.newaxis], wy[np.newaxis,:] )
    
    w  = np.matmul(wxy[:,:, np.newaxis], wz[np.newaxis, :])
    
    return w




def seg_2_bounding_box(y, index):
    
    if np.mean(y==0)+ np.mean(y==1)<1:
        
        print('The segmentation mask must include ones and zeros only!    Returning NaN. ', index)
        return np.nan
    
    if y.shape[-1]!=2:
        
        print('Input segmentation mask must have two channels!   Returning NaN. ', index)
        return np.nan
    
    z = np.where(y[:,:,:,1] > 0.5)
    x_min, x_max, y_min, y_max, z_min, z_max = z[0].min(), z[0].max(), z[1].min(), z[1].max(), z[2].min(), z[2].max()
    
    yt= np.zeros( y.shape[:-1] )
    yt[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]= 1
    
    yn= np.zeros( y.shape )
    yn[:, :, :, 1] = yt
    yn[:, :, :, 0] = 1-yt
    
    return yn


def save_image_and_maskt_humbs(x_i, y_i, x_f, y_f, vol_name, thumbs_dir):

    z_i = np.where(y_i > 0)
    z_f = np.where(y_f > 0)

    if len(z_i[0]) < 10:

        print('Segmentation mask smaller than 10; image index: ') #, image_index)

    else:

        n_rows, n_cols = 3, 4

        fig, ax = plt.subplots(figsize=(20, 13), nrows=n_rows, ncols=n_cols)

        x_min, x_max, y_min, y_max, z_min, z_max = z_i[0].min(), z_i[0].max(), z_i[1].min(), z_i[1].max(), z_i[2].min(), z_i[ 2].max()

        SX = (x_min + x_max)
        SY = (y_min + y_max)
        SZ = (z_min + z_max)

        plt.subplot(n_rows, n_cols, 1)
        plt.imshow(x_i[:, :, SZ // 2], cmap='gray')
        plt.subplot(n_rows, n_cols, 5)
        plt.imshow(x_i[:, SY // 2, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 9)
        plt.imshow(x_i[SX // 2, :, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 2)
        plt.imshow(y_i[:, :, SZ // 2], cmap='gray')
        plt.subplot(n_rows, n_cols, 6)
        plt.imshow(y_i[:, SY // 2, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 10)
        plt.imshow(y_i[SX // 2, :, :], cmap='gray')

        x_min, x_max, y_min, y_max, z_min, z_max = z_f[0].min(), z_f[0].max(), z_f[1].min(), z_f[1].max(), z_f[2].min(), z_f[2].max()

        SX = (x_min + x_max)
        SY = (y_min + y_max)
        SZ = (z_min + z_max)

        plt.subplot(n_rows, n_cols, 3)
        plt.imshow(x_f[:, :, SZ // 2], cmap='gray')
        plt.subplot(n_rows, n_cols, 7)
        plt.imshow(x_f[:, SY // 2, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 11)
        plt.imshow(x_f[SX // 2, :, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 4)
        plt.imshow(y_f[:, :, SZ // 2], cmap='gray')
        plt.subplot(n_rows, n_cols, 8)
        plt.imshow(y_f[:, SY // 2, :], cmap='gray')
        plt.subplot(n_rows, n_cols, 12)
        plt.imshow(y_f[SX // 2, :, :], cmap='gray')

    fig.savefig(thumbs_dir + vol_name + '.png')
    plt.close(fig)




def create_rough_mask_old(x, percentile=95, closing_rad=3):
    
    seg= np.uint(x>np.nanpercentile(x, percentile))
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    
    seg = sitk.GetImageFromArray(seg)
    seg = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg)
    
    n_component = seg.max()
    
    largest_component= -1
    largest_size= -1
    
    for i in range(1, n_component + 1):
        
        if np.sum(seg == i) > largest_size:
            largest_component= i
            largest_size= np.sum(seg==i)
            
    if largest_component==-1:
        print('Segmentation empty! Returning NaN')
        seg= np.nan
    else:
        seg= np.uint(seg==largest_component)
        seg = sitk.GetImageFromArray(seg)
        seg = sitk.BinaryMorphologicalClosing(seg, closing_rad)
        seg = sitk.GetArrayFromImage(seg)
        
    return seg




def create_rough_mask(x, percentile=95, closing_rad=3):
    
    seg= np.uint(x>np.nanpercentile(x, percentile))
    
    seg = sitk.GetImageFromArray(seg)
    
    seg = sitk.BinaryMorphologicalClosing(seg, 1)
    
    seg = sitk.BinaryMorphologicalOpening(seg, 1)
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    seg = c_filter.Execute(seg)
    
    seg = sitk.GetArrayFromImage(seg)
    
    n_component = seg.max()
    
    largest_component= -1
    largest_size= -1
    
    for i in range(1, n_component + 1):
        
        if np.sum(seg == i) > largest_size:
            largest_component= i
            largest_size= np.sum(seg==i)
            
    if largest_component==-1:
        print('Segmentation empty! Returning NaN')
        seg= np.nan
    else:
        seg= np.uint(seg==largest_component)
        seg = sitk.GetImageFromArray(seg)
        seg = sitk.BinaryMorphologicalClosing(seg, closing_rad)
        seg = sitk.GetArrayFromImage(seg)
        
    return seg



def create_rough_mask_staple(x, y, percentile_vector=np.array([93, 95, 98]), closing_rad_vector=np.array([15,20,30])):
    
    segmentations= list()
    perfs= list()
    
    for percentile in percentile_vector:
        for closing_rad in closing_rad_vector:
            
            seg= np.uint(x>np.nanpercentile(x, percentile))
            
            c_filter = sitk.ConnectedComponentImageFilter()
            c_filter.FullyConnectedOn()
            
            seg = sitk.GetImageFromArray(seg)
            seg = c_filter.Execute(seg)
            seg = sitk.GetArrayFromImage(seg)
            
            n_component = seg.max()
            
            largest_component= -1
            largest_size= -1
            
            for i in range(1, n_component + 1):
                
                if np.sum(seg == i) > largest_size:
                    largest_component= i
                    largest_size= np.sum(seg==i)
                    
            if largest_component==-1:
                print('Segmentation empty!')
            else:
                seg= np.uint(seg==largest_component)
                seg = sitk.GetImageFromArray(seg)
                seg = sitk.BinaryMorphologicalClosing(seg, int(closing_rad))
                segmentations.append(seg)
                temp= sitk.GetArrayFromImage(seg)
                perfs.append(dk_seg.dice(y, temp))
    
    foregroundValue = 1
    threshold = 0.95
    seg_staple = sitk.STAPLE(segmentations, foregroundValue) > threshold
    
    seg= sitk.GetArrayFromImage(seg_staple)
    
    return seg, perfs
    


def empty_folder(this_folder):
    
    for file in os.listdir(this_folder):
        file_path = os.path.join(this_folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)




def save_img_slice_and_seg_boundary(x, y, slc_no, vol_name, thumbs_dir, y2, slc_in=2, markersize=1):

    b= dk_seg.seg_2_boundary_3d(y)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==slc_no
    x_sel= z[0][slc_sel].astype(np.int)
    y_sel= z[1][slc_sel].astype(np.int)
        
    b= dk_seg.seg_2_boundary_3d(y2)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==slc_no
    x_sel2= z[0][slc_sel].astype(np.int)
    y_sel2= z[1][slc_sel].astype(np.int)
    
    fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1)
    
    plt.imshow(x[:,:,slc_no], cmap='gray')
    
    plt.plot(y_sel, x_sel, '.r',  markersize=markersize)
    plt.plot(y_sel2, x_sel2, '.b',  markersize=markersize)
    
    fig.savefig(thumbs_dir + vol_name + '.png')
    plt.close(fig)
    
    



def compute_seg_vol_to_surface_ratio(y):
    
    b= dk_seg.seg_2_boundary_3d(y>0.5)
    
    vol= np.sum(y>0.5)
    
    surf= np.sum(b)
    
    return vol/surf



def compute_seg_vol_to_diamater_ratio(y):
    
    b= dk_seg.seg_2_boundary_3d(y>0.5)
    b= np.where(b>0)
    b= np.vstack((b[0], b[1], b[2])).T
    
    D = pdist(b)
    D = squareform(D)
    diameter= D.max()
    
    vol= np.sum(y>0.5)
    
    return vol, diameter, vol/diameter





def weight_matrix(lx, ly, lz, sigx, sigy, sigz, n_channel=1):
    
    wx= np.exp( - ( np.arange(1, lx+1)-lx/2 )**2/ sigx**2 )
    wy= np.exp( - ( np.arange(1, ly+1)-ly/2 )**2/ sigy**2 )
    wz= np.exp( - ( np.arange(1, lz+1)-lz/2 )**2/ sigz**2 )
    
    wxy= np.matmul( wx[:,np.newaxis] , wy[np.newaxis,:] )
    wxyz= np.matmul( wxy[:,:,np.newaxis] , wz[np.newaxis,:] )
    
    if n_channel==1:
        W= wxyz
    else:
        W= np.zeros((lx,ly,lz,n_channel))
        for i_channel in range(n_channel):
            W[:,:,:,i_channel]= wxyz.copy()
    
    return W





def save_multiscale_thumbs( my_list, slice_ind=-1, direction= 'Z', image_index=None, iter_num=-1, save_dir= None, figsize=(22, 13)):
    
    n_img= len(my_list)
    
    n_rows= int(np.sqrt(n_img))
    n_cols= n_rows
    while n_rows*n_cols<n_img:
        n_cols+=1
    
    if direction=='X':
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [2,1,0] )
    elif direction=='Y':
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [0,2,1] )
    elif direction=='Z':
        pass
    else:
        print('Direction not valid')
        return None
    
    if slice_ind==-1:
        y_t= my_list[1]
        z = np.where(y_t > 0)
        SZ = ( z[2].min() + z[2].max() ) // 2
        
    fig, ax = plt.subplots(figsize=figsize, nrows=n_rows, ncols=n_cols)
    
    for i in range(len(my_list)):
        plt.subplot(n_rows, n_cols, i+1)
        if i==0:
            plt.imshow( my_list[i] [:, :, SZ ], cmap='gray')
        else:
            plt.imshow( my_list[i] [:, :, SZ ])
        '''elif i==1:
            plt.imshow( my_list[i] [:, :, SZ ])
        else:
            plt.imshow( my_list[i] [:, :, SZ ]-my_list[1] [:, :, SZ ])'''
    
    plt.tight_layout()
    
    if iter_num>-1:
        fig.savefig(save_dir + 'thumbs_' + direction + '_' + str(image_index) + '_' + str(iter_num) + '.png')
    else:
        fig.savefig(save_dir + 'thumbs_' + direction + '_' + str(image_index) + '.png')
    
    plt.close(fig)






def save_cae_thumbs(x_c, x_r, training_flag, image_index, iteration_count, prediction_dir):

    n_rows, n_cols = 2, 3

    SX, SY, SZ = x_c.shape

    fig, ax = plt.subplots(figsize=(16, 10), nrows=n_rows, ncols=n_cols)

    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(x_c[:, :, SZ // 2], cmap='gray')
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(x_c[:, SY // 2, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(x_c[SX // 2, :, :], cmap='gray')
    
    plt.subplot(n_rows, n_cols, 4)
    plt.imshow(x_r[:, :, SZ // 2], cmap='gray')
    plt.subplot(n_rows, n_cols, 5)
    plt.imshow(x_r[:, SY // 2, :], cmap='gray')
    plt.subplot(n_rows, n_cols, 6)
    plt.imshow(x_r[SX // 2, :, :], cmap='gray')
    
    if training_flag:
        fig.savefig(prediction_dir + 'X_train_' + str(image_index) + '_' + str(iteration_count) + '.png')
    else:
        fig.savefig(prediction_dir + 'X_test_' + str(image_index) + '_' + str(iteration_count) + '.png')
    
    plt.close(fig)






def save_dwi_rough_thumbs( my_list, slice_ind=-1, direction= 'Z', image_index=None, save_dir= None):
    
    n_img= len(my_list)
    n_rows, n_cols = 2, (n_img+1)//2
    
    if direction=='X':
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [2,1,0] )
    elif direction=='Y':
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [0,2,1] )
    elif direction=='Z':
        pass
    else:
        print('Direction not valid')
        return None
    
    if slice_ind==-1:
        y_t= my_list[1]
        z = np.where(y_t > 0)
        SZ = ( z[2].min() + z[2].max() ) // 2
        
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    for i in range(len(my_list)):
        plt.subplot(n_rows, n_cols, i+1)
        if i==0:
            plt.imshow( my_list[i] [:, :, SZ ], cmap='gray')
        else:
            plt.imshow( my_list[i] [:, :, SZ ])
    
    plt.tight_layout()
    
    fig.savefig(save_dir + 'DWI_' + direction + '_' + str(image_index) + '.png')
    
    plt.close(fig)






def save_dwi_rough_boundaries( my_list, slice_ind=-1, direction= 'Z', image_index=None, save_dir= None, figsize=(8, 8), markersize=1, colors=['.r', '.b', '.g', '.c', '.y', '.k']):
    
    n_rows, n_cols = 1, 1
    
    if direction=='X':
        slc_in= 0
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [2,1,0] )
    elif direction=='Y':
        slc_in= 1
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [0,2,1] )
    elif direction=='Z':
        slc_in= 2
        pass
    else:
        print('Direction not valid')
        return None
    
    if slice_ind==-1:
        y_t= my_list[1]
        z = np.where(y_t > 0)
        SZ = ( z[2].min() + z[2].max() ) // 2
        
    fig, ax = plt.subplots(figsize=figsize, nrows=n_rows, ncols=n_cols)
    
    plt.imshow( my_list[0] [:, :, SZ ], cmap='gray')
    plt.axis('off');
    
    for i in range(2, len(my_list)):
        
        y_temp= my_list[i]
        
        b= dk_seg.seg_2_boundary_3d(y_temp)
        z= np.where(b>0)
        slc_all= z[slc_in].astype(np.int)
        slc_sel= slc_all==SZ
        x_sel= z[0][slc_sel].astype(np.int)
        y_sel= z[1][slc_sel].astype(np.int)
        plt.plot(y_sel, x_sel, colors[i-1],  markersize=markersize)
    
    i= 1
    
    y_temp= my_list[i]
    
    b= dk_seg.seg_2_boundary_3d(y_temp)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==SZ
    x_sel= z[0][slc_sel].astype(np.int)
    y_sel= z[1][slc_sel].astype(np.int)
    plt.plot(y_sel, x_sel, colors[i-1],  markersize=markersize)
    
    fig.savefig(save_dir + 'DWI_boundary_' + direction + '_' + str(image_index) + '.png')
    
    plt.close(fig)











                
def save_co_teaching_boundaries( my_list, slice_ind=-1, direction= 'Z', image_index=None, eval_index= None, train=False, save_dir= None, 
                                figsize=(8, 8), markersize=1, colors=['.r', '.b', '.g', '.c', '.y', '.k']):
    
    n_rows, n_cols = 1, 1
    
    if direction=='X':
        slc_in= 0
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [2,1,0] )
    elif direction=='Y':
        slc_in= 1
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [0,2,1] )
    elif direction=='Z':
        slc_in= 2
        pass
    else:
        print('Direction not valid')
        return None
    
    if slice_ind==-1:
        y_t= my_list[1]
        z = np.where(y_t > 0)
        SZ = ( z[2].min() + z[2].max() ) // 2
        
    fig, ax = plt.subplots(figsize=figsize, nrows=n_rows, ncols=n_cols)
    
    plt.imshow( my_list[0] [:, :, SZ ], cmap='gray')
    plt.axis('off');
    
    for i in range(2, len(my_list)):
        
        y_temp= my_list[i]
        
        b= dk_seg.seg_2_boundary_3d(y_temp)
        z= np.where(b>0)
        slc_all= z[slc_in].astype(np.int)
        slc_sel= slc_all==SZ
        x_sel= z[0][slc_sel].astype(np.int)
        y_sel= z[1][slc_sel].astype(np.int)
        plt.plot(y_sel, x_sel, colors[i-1],  markersize=markersize)
    
    i= 1
    
    y_temp= my_list[i]
    
    b= dk_seg.seg_2_boundary_3d(y_temp)
    z= np.where(b>0)
    slc_all= z[slc_in].astype(np.int)
    slc_sel= slc_all==SZ
    x_sel= z[0][slc_sel].astype(np.int)
    y_sel= z[1][slc_sel].astype(np.int)
    plt.plot(y_sel, x_sel, colors[i-1],  markersize=markersize)
    
    if train:
        fig.savefig(save_dir + 'X_ct_train_' + direction + '_' + str(image_index) + '_' + str(eval_index) + '.png')
    else:
        fig.savefig(save_dir + 'X_ct_test_' + direction + '_' + str(image_index) + '_' + str(eval_index) + '.png')
    
    plt.close(fig)







def resample_imtar_to_imref(im_tar, im_ref, resampling_method= sitk.sitkLinear, match_ref_pixeltype=False):
    
    if match_ref_pixeltype:
        pixeltype= im_ref.GetPixelIDValue()
    else:
        pixeltype= im_tar.GetPixelIDValue()
    
    I = sitk.Image(im_ref.GetSize(), pixeltype)
    I.SetSpacing(im_ref.GetSpacing())
    I.SetOrigin(im_ref.GetOrigin())
    I.SetDirection(im_ref.GetDirection())
    
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(I)
    resample.SetInterpolator( resampling_method )
    resample.SetTransform(sitk.Transform())
    
    I = resample.Execute(im_tar)
    
    return I
    






def save_dwi_gold_boundaries( my_list, image_index=None, save_dir= None, markersize=1, colors=['.r', '.b']):
    
    n_rows, n_cols = 1, 3
    
    fig, ax = plt.subplots(figsize=(18,8), nrows=n_rows, ncols=n_cols)
    i_fig= 0
    
    for direction in ['Z', 'X', 'Y']:
        
        if direction=='X':
            slc_in= 2
            for i in range(len(my_list)):
                my_list[i]= np.transpose( my_list[i] , [2,1,0] )
        elif direction=='Y':
            slc_in= 2
            for i in range(len(my_list)):
                my_list[i]= np.transpose( my_list[i] , [0,2,1] )
        elif direction=='Z':
            slc_in= 2
            pass
        
        y_t= my_list[1]
        z = np.where(y_t > 0)
        SZ = ( z[2].min() + z[2].max() ) // 2
        
        i_fig+= 1
        plt.subplot(n_rows, n_cols, i_fig)
        
        plt.imshow( my_list[0] [:, :, SZ ], cmap='gray')
        plt.axis('off');
        
        for i in range(1, len(my_list)):
            
            y_temp= my_list[i]
            
            b= dk_seg.seg_2_boundary_3d(y_temp)
            z= np.where(b>0)
            slc_all= z[slc_in].astype(np.int)
            slc_sel= slc_all==SZ
            x_sel= z[0][slc_sel].astype(np.int)
            y_sel= z[1][slc_sel].astype(np.int)
            plt.plot(y_sel, x_sel, colors[i-1],  markersize=markersize)
    
    plt.tight_layout()
    
    fig.savefig(save_dir + 'X_dwi_seg_gold_' + str(image_index) + '.png')
    
    plt.close(fig)













def number_n_size_of_cc(seg):
    
    c_filter = sitk.ConnectedComponentImageFilter()
    c_filter.FullyConnectedOn()
    
    seg = sitk.GetImageFromArray(seg)
    seg = c_filter.Execute(seg)
    seg = sitk.GetArrayFromImage(seg)
    
    n_component = seg.max()
    
    size= np.zeros(n_component)
    
    for i in range(n_component):
        
        size[i]= np.sum(seg==i+1)
    
    return n_component, size

def save_data_thumbs( my_list, slice_ind=-1, direction= 'Z', image_index=None, save_dir= None):
    
    n_img= len(my_list)
    n_rows, n_cols = 2, (n_img+1)//2
    
    if direction=='X':
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [2,1,0] )
    elif direction=='Y':
        for i in range(len(my_list)):
            my_list[i]= np.transpose( my_list[i] , [0,2,1] )
    elif direction=='Z':
        pass
    else:
        print('Direction not valid')
        return None
    
    if slice_ind==-1:
        y_t= my_list[1]
        z = np.where(y_t > 0)
        SZ = ( z[2].min() + z[2].max() ) // 2
        
    fig, ax = plt.subplots(figsize=(22, 13), nrows=n_rows, ncols=n_cols)
    
    for i in range(len(my_list)):
        plt.subplot(n_rows, n_cols, i+1)
        if i==0:
            plt.imshow( my_list[i] [:, :, SZ ], cmap='gray')
        elif i==1:
            plt.imshow( my_list[i] [:, :, SZ ])
        else:
            plt.imshow( my_list[i] [:, :, SZ ]-my_list[1] [:, :, SZ ])
    
    fig.savefig(save_dir + 'CPSP_' + direction + '_' + str(image_index) + '.png')
    
    plt.close(fig)




def nbr_sum_6(x):
    
    if np.mean(x==1)+np.mean(x==0)==0:
        
        print('The passed array is not binary,   returning NaN')
        y= np.nan
        
    else:
        
        a, b, c= x.shape
        
        y= np.zeros(x.shape)
        z= np.nonzero(x)
        
        if len(z[0])>1:
            y= np.zeros(x.shape)
            shift_x, shift_y, shift_z= 1, 0, 0
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= -1, 0, 0
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= 0, 1, 0
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= 0, -1, 0
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= 0, 0, 1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= 0, 0, -1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
                        
    return y



def nbr_sum_8(x):
    
    if np.mean(x==1)+np.mean(x==0)==0:
        
        print('The passed array is not binary,   returning NaN')
        y= np.nan
        
    else:
        
        a, b, c= x.shape
        
        y= np.zeros(x.shape)
        z= np.nonzero(x)
        
        if len(z[0])>1:
            y= np.zeros(x.shape)
            shift_x, shift_y, shift_z= 1, 1, 1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= 1, 1, -1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= 1, -1, 1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= 1, -1, -1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= -1, 1, 1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= -1, 1, -1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= -1, -1, 1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            shift_x, shift_y, shift_z= -1, -1, -1
            y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
                       
    return y






def nbr_sum_26(x):
    
    if np.mean(x==1)+np.mean(x==0)==0:
        
        print('The passed array is not binary,   returning NaN')
        y= np.nan
        
    else:
        
        a, b, c= x.shape
        
        y= np.zeros(x.shape)
        z= np.nonzero(x)
        
        if len(z[0])>1:
            y= np.zeros(x.shape)
            for shift_x in range(-1, 2):
                for shift_y in range(-1, 2):
                    for shift_z in range(-1, 2):
                        y[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
            
            y-= x
            
    return y


def nbr_sum_124(x):
    
    if np.mean(x==1)+np.mean(x==0)==0:
        
        print('The passed array is not binary,   returning NaN')
        y= np.nan
        
    else:
        
        a, b, c= x.shape
        
        y= np.zeros(x.shape)
        z= np.nonzero(x)
        
        if len(z[0])>1:
            y= np.zeros(x.shape)
            for shift_x in range(-2, 3):
                for shift_y in range(-2, 3):
                    for shift_z in range(-2, 3):
                        y[2:-2,2:-2,2:-2]+= x[2+shift_x:a-2+shift_x, 2+shift_y:b-2+shift_y, 2+shift_z:c-2+shift_z]
            
            y-= x
            
    return y





def estimate_ECE_and_MCE(seg_true, seg_pred, N=10, plot_save_path=''):
    
    ECE_curve= np.zeros((N,3))
    
    for i in range(N):
        
        p0= i/N
        p1= (i+1)/N
        
        mask= np.logical_and(seg_pred>p0, seg_pred<p1)
        
        pos= mask*seg_true
        
        mean_p= seg_pred[mask].mean()
        mean_q= pos.sum()/ mask.sum()
        frac =  mask.mean()
        
        ECE_curve[i,:]= mean_p, mean_q, frac
        
    ECE= np.sum(ECE_curve[:,2] * np.abs(ECE_curve[:,0]- ECE_curve[:,1] ))
    MCE= np.abs(ECE_curve[:,0]- ECE_curve[:,1] ).max()
    
    if not plot_save_path is None:
        fig, ax = plt.subplots(figsize=(12, 12), nrows=1, ncols=1)
        plt.subplot(1, 1, 1)
        plt.plot( ECE_curve[:,0], ECE_curve[:,1], '.')
        fig.savefig(plot_save_path)
        plt.close(fig)
    
    return ECE, MCE, ECE_curve








def estimate_ECE_and_MCE_masked(seg_true, seg_pred, error_mask, N=10, plot_save_path=''):
    
    ECE_curve= np.zeros((N,3))
    
    for i in range(N):
        
        p0= i/N
        p1= (i+1)/N
        
        mask= np.logical_and( np.logical_and(seg_pred>p0, seg_pred<p1), error_mask )
        
        pos= mask*seg_true
        
        mean_p= seg_pred[mask].mean()
        mean_q= pos.sum()/ mask.sum()
        frac =  mask.mean()
        
        ECE_curve[i,:]= mean_p, mean_q, frac
        
    frac_sum= np.sum(ECE_curve[:,2])
    ECE_curve[:,2]/= frac_sum
    
    ECE= np.sum(ECE_curve[:,2] * np.abs(ECE_curve[:,0]- ECE_curve[:,1] ))
    MCE= np.abs(ECE_curve[:,0]- ECE_curve[:,1] ).max()
    
    if not plot_save_path is None:
        fig, ax = plt.subplots(figsize=(12, 12), nrows=1, ncols=1)
        plt.subplot(1, 1, 1)
        plt.plot( ECE_curve[:,0], ECE_curve[:,1], '.')
        fig.savefig(plot_save_path)
        plt.close(fig)
    
    return ECE, MCE, ECE_curve









def seg_2_boundary_3d(x):
    
    a, b, c= x.shape
    
    y= np.zeros(x.shape)
    z= np.nonzero(x)
    
    if len(z[0])>1:
        x_sum= np.zeros(x.shape)
        for shift_x in range(-1, 2):
            for shift_y in range(-1, 2):
                for shift_z in range(-1, 2):
                    x_sum[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
        y= np.logical_and( x==1 , np.logical_and( x_sum>0, x_sum<27 ) )
        
    return y



def seg_2_anulus(mask_orig, radius= 2.0):
    
    mask_copy= mask_orig.copy()
    
    size_x, size_y, size_z= mask_copy.shape
    mask= np.zeros((size_x+20, size_y+20, size_z+20))
    mask[10:10+size_x, 10:10+size_y, 10:10+size_z]= mask_copy
    
    mask_boundary= seg_2_boundary_3d(mask)
    
    mask_boundary= sitk.GetImageFromArray(mask_boundary.astype(np.int))
    
    dist_image = sitk.SignedMaurerDistanceMap(mask_boundary, insideIsPositive=True, useImageSpacing=True,
                                           squaredDistance=False)
    
    dist_image= - sitk.GetArrayFromImage(dist_image)
    dist_image= dist_image<radius
    
    anulus= dist_image
    
    anulus= anulus[10:10+size_x, 10:10+size_y, 10:10+size_z]
    
    return anulus

















def register_jhu(my_t2, my_mk, jh_t2, jh_mk, jh_lb):
    
    my_t2_np= sitk.GetArrayFromImage( my_t2)
    my_mk_np= sitk.GetArrayFromImage( my_mk)
    
    my_t2_mk_np= my_t2_np * my_mk_np
    my_t2_mk= sitk.GetImageFromArray(my_t2_mk_np)
    
    my_t2_mk.SetDirection(my_mk.GetDirection())
    my_t2_mk.SetOrigin(my_mk.GetOrigin())
    my_t2_mk.SetSpacing(my_mk.GetSpacing())
    
    fixed_image= my_t2_mk
    
    jh_t2_np= sitk.GetArrayFromImage( jh_t2)
    jh_mk_np= sitk.GetArrayFromImage( jh_mk)
    
    jh_t2_mk_np= jh_t2_np * (jh_mk_np>200)
    jh_t2_mk= sitk.GetImageFromArray(jh_t2_mk_np)
    
    jh_t2_mk.SetDirection(jh_mk.GetDirection())
    jh_t2_mk.SetOrigin(jh_mk.GetOrigin())
    jh_t2_mk.SetSpacing(jh_mk.GetSpacing())
    
    moving_image= jh_t2_mk
    
    moving_image.SetDirection( fixed_image.GetDirection() )
    jh_lb.SetDirection( fixed_image.GetDirection() )
    
    initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image,moving_image.GetPixelID()), 
                                                          moving_image, 
                                                          sitk.Euler3DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    resample.SetInterpolator(sitk.sitkLinear)  
    resample.SetTransform(initial_transform)
    
    moving_image_2= resample.Execute(moving_image)
    
    registration_method = sitk.ImageRegistrationMethod()
    
    grid_physical_spacing = [50.0, 50.0, 50.0]
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]
    
    final_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, 
                                                         transformDomainMeshSize = mesh_size, order=3)    
    registration_method.SetInitialTransform(final_transform)
    
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)
    
    registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                sitk.Cast(moving_image_2, sitk.sitkFloat32))
    
    final_transform_v = sitk.Transform(final_transform)
    
    '''resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    resample.SetInterpolator(sitk.sitkLinear)  
    resample.SetTransform(final_transform_v)
    
    moving_image_5= resample.Execute(moving_image_2)
    sitk.WriteImage(moving_image_5 , reg_dir+'moving_image_reg.mhd')'''
    
    final_transform_v.AddTransform(initial_transform)
    
    '''resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    resample.SetInterpolator(sitk.sitkNearestNeighbor)  
    resample.SetTransform(final_transform_v)
    
    moving_image_5= resample.Execute(jh_lb)
    sitk.WriteImage(moving_image_5 , reg_dir+'lb_image_reg.mhd')'''
        
    tx= initial_transform
    tx.AddTransform(final_transform)
    
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    resample.SetInterpolator(sitk.sitkNearestNeighbor)  
    resample.SetTransform(tx)
    
    out_image= resample.Execute(jh_lb)
    
    return out_image






























def save_all_data_thumbs( X_vol, Y_vol, save_dir= None):
    
    n_vol, SX, SY, SZ, n_channel= X_vol.shape
    n_vol, SX, SY, SZ, n_class  = Y_vol.shape
    
    n_rows, n_cols = 1, n_channel+1
    
    for i_vol in range(n_vol):
        
        fig, ax = plt.subplots(figsize=(18,8), nrows=n_rows, ncols=n_cols)
        i_fig= 0
        
        for i_channel in range(n_channel):
            
            i_fig+= 1
            plt.subplot(n_rows, n_cols, i_fig)
            
            plt.imshow( X_vol[i_vol, :, :, SZ//2 , i_channel], cmap='gray')
            plt.axis('off');
            
        seg= np.zeros( (SX, SY, SZ) )
        
        for i_seg in range(1,n_class):
            
            seg+= i_seg*Y_vol[i_vol,:,:,:,i_seg]
        
        i_fig+= 1
        plt.subplot(n_rows, n_cols, i_fig)
        plt.imshow( seg[:, :, SZ//2], vmin=0, vmax= n_class)
        plt.axis('off');
        
        plt.tight_layout()
        
        fig.savefig(save_dir + 'thumb_data' + str(i_vol) + '.png')
        
        plt.close(fig)






def compute_AUC(D_te, D_od, p):
    
    fpr= np.zeros(len(p))
    tpr= np.zeros(len(p))
    
    for ip in range(len(p)):
        
        fpr[ip]= np.mean(D_te>p[ip])
        tpr[ip]= np.mean(D_od>p[ip])
    
    return metrics.auc(fpr, tpr), fpr, tpr
    















def save_img_and_seg_boundaries( my_list_orig, image_index=None, save_dir= None, markersize=1, colors=['.r', '.b']):
    
    n_rows, n_cols = 1, 3
    
    fig, ax = plt.subplots(figsize=(18,8), nrows=n_rows, ncols=n_cols)
    i_fig= 0
    my_list= [None]*len(my_list_orig)
    
    for direction in ['Z', 'X', 'Y']:
        
        if direction=='X':
            slc_in= 2
            for i in range(len(my_list)):
                my_list[i]= np.transpose( my_list_orig[i] , [2,1,0] )
        elif direction=='Y':
            slc_in= 2
            for i in range(len(my_list)):
                my_list[i]= np.transpose( my_list_orig[i] , [0,2,1] )
        elif direction=='Z':
            slc_in= 2
            for i in range(len(my_list)):
                my_list[i]= my_list_orig[i]
            pass
        
        y_t= my_list[1]
        z = np.where(y_t > 0)
        SZ = ( z[2].min() + z[2].max() ) // 2
        
        i_fig+= 1
        plt.subplot(n_rows, n_cols, i_fig)
        
        plt.imshow( my_list[0] [:, :, SZ ], cmap='gray')
        plt.axis('off');
        
        for i in range(1, len(my_list)):
            
            y_temp= my_list[i]
            
            b= dk_seg.seg_2_boundary_3d(y_temp)
            z= np.where(b>0)
            slc_all= z[slc_in].astype(np.int)
            slc_sel= slc_all==SZ
            x_sel= z[0][slc_sel].astype(np.int)
            y_sel= z[1][slc_sel].astype(np.int)
            plt.plot(y_sel, x_sel, colors[i-1],  markersize=markersize)
    
    plt.tight_layout()
    
    fig.savefig(save_dir + 'X_dwi_seg_gold_' + str(image_index) + '.png')
    
    plt.close(fig)












