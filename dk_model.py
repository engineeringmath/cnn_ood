# -*- coding: utf-8 -*-
"""

Models for segmentation

@author: davoo
"""


import numpy as np
import tensorflow as tf



def slice_and_scale(tensor, image_idx=0, slice_idx=0, channel_idx=0):
    t = tensor[image_idx, :, :, slice_idx, channel_idx]
    sx, sy= t.shape
    t = (t - tf.reduce_min(t)) / tf.reduce_max(t) * 255
    t = tf.reshape(t, tf.stack((1, sx, sy, 1)))
    return t


def davood_net(X, ks_0, depth, n_feat_0, n_channel, n_class, p_keep_conv, bias_init=0.001, write_summaries= False):
    
    feat_fine = [None] * (depth - 1)
    
    if write_summaries:
        tf.summary.image('input image', slice_and_scale(X))
    
    for level in range(depth):
        
        ks = ks_0
                
        if level == 0:
            
            strd = 1
            
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            name_c = 'Conv_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
        else:
            
            strd = 2
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            name_c = 'Conv_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            for i in range(1, level):
                n_l = n_feat_0 * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                name_w = 'W_' + str(level) + '_' + str(i) + '_init'
                name_b = 'b_' + str(level) + '_' + str(i) + '_init'
                name_c = 'Conv_' + str(level) + '_' + str(i) + '_init'
                W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat_0, n_feat_0], stddev=s_dev), name=name_w)
                b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
                inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                inp= tf.nn.dropout(inp, p_keep_conv)
                
            for level_reg in range(0, level):
                
                inp_0 = feat_fine[level_reg]
                
                level_diff = level - level_reg
                
                n_feat = n_feat_0 * 2 ** level_reg
                n_l = n_feat * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                
                for j in range(level_diff):
                    name_w = 'W_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    name_b = 'b_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    name_c = 'Conv_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
                    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
                    inp_0 = tf.nn.relu(
                        tf.add(tf.nn.conv3d(inp_0, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                    inp_0 = tf.nn.dropout(inp_0, p_keep_conv)
                    
                inp = tf.concat([inp, inp_0], 4)
                
        ks = ks_0
        
        n_feat = n_feat_0 * 2 ** level
        
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_down'
            name_b = 'b_' + str(level) + '_2_down'
            name_c = 'Conv_' + str(level) + '_2_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_down'
            name_b = 'b_' + str(level) + '_3_down'
            name_c = 'Conv_' + str(level) + '_3_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_down'
            name_b = 'b_' + str(level) + '_4_down'
            name_c = 'Conv_' + str(level) + '_4_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_down'
            name_b = 'b_' + str(level) + '_5_down'
            name_c = 'Conv_' + str(level) + '_5_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_1 + inp_0  ###
            
        if level < depth - 1:
            feat_fine[level] = inp
            
    # DeConvolution Layers
    
    for level in range(depth - 2, -1, -1):
        
        ks = ks_0
        
        n_l = n_feat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_up'
        name_b = 'b_' + str(level) + '_up'
        name_c = 'Conv_' + str(level) + '_up'
        W_deconv = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat // 2, n_feat], stddev=s_dev), name=name_w)
        b_deconv = tf.Variable(tf.constant(bias_init, shape=[n_feat // 2]), name=name_b)
        in_shape = tf.shape(inp)
        if level == 3:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, 9, in_shape[4] // 2])
        else:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, in_shape[4] // 2])
        Deconv = tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
        Deconv = tf.nn.relu(tf.add(Deconv, b_deconv))
        Deconv= tf.nn.dropout(Deconv, p_keep_conv)
        inp = tf.concat([feat_fine[level], Deconv], 4)
        
        if write_summaries:
            tf.summary.histogram(name_w, W_deconv)
            tf.summary.histogram(name_b, b_deconv)
            tf.summary.image(name_c, slice_and_scale(inp))
            
        if level == depth - 2:
            n_concat = n_feat
        else:
            n_concat = n_feat * 3 // 4
            
        if level < depth - 2:
            n_feat = n_feat // 2
            
        n_l = n_concat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_1_up'
        name_b = 'b_' + str(level) + '_1_up'
        name_c = 'Conv_' + str(level) + '_1_up'
        W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_concat, n_feat], stddev=s_dev), name=name_w)
        b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
        inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1))
        inp= tf.nn.dropout(inp, p_keep_conv)
        
        if write_summaries:
            tf.summary.histogram(name_w, W_1)
            tf.summary.histogram(name_b, b_1)
            tf.summary.image(name_c, slice_and_scale(inp))
            
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_up'
            name_b = 'b_' + str(level) + '_2_up'
            name_c = 'Conv_' + str(level) + '_2_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_up'
            name_b = 'b_' + str(level) + '_3_up'
            name_c = 'Conv_' + str(level) + '_3_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_up'
            name_b = 'b_' + str(level) + '_4_up'
            name_c = 'Conv_' + str(level) + '_4_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_up'
            name_b = 'b_' + str(level) + '_5_up'
            name_c = 'Conv_' + str(level) + '_5_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_1 + inp_0  ###
            
     #        if level==0:
     #            n_l= n_feat*ks**3
     #            s_dev= np.sqrt(2.0/n_l)
     #            name_w= 'W_up'
     #            name_b= 'b_up'
     #            name_c= 'Conv_up'
     #            W_deconv= tf.Variable(tf.truncated_normal([ks,ks,ks,n_class,n_feat], stddev=s_dev), name=name_w)
     #            b_deconv= tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
     #            in_shape = tf.shape(inp)
     #            out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]*2, n_class])
     #            Deconv= tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1,2,2,2,1], padding='SAME')
     #            output= tf.add(Deconv, b_deconv)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w = 'W_out'
    name_b = 'b_out'
#    name_c = 'Conv_out'
    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_class], stddev=s_dev), name=name_w)
    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
    output = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1)
    # output= tf.nn.dropout(output, p_keep_conv)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w_s = 'W_out_s'
    name_b_s = 'b_out_s'
#    name_c_s = 'Conv_out_s'
    W_1_s = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, 1], stddev=s_dev), name=name_w_s)
    b_1_s = tf.Variable(tf.constant(bias_init, shape=[1]), name=name_b_s)
    output_s = tf.add(tf.nn.conv3d(inp, W_1_s, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1_s)
    # output= tf.nn.dropout(output, p_keep_conv)
    
    '''if write_summaries:
        tf.summary.histogram(name_w, W_1)
        tf.summary.histogram(name_b, b_1)
        tf.summary.image(name_c, slice_and_scale(inp))'''
        
    return output, output_s







def davood_3d_cae(X, ks_0, n_feat_vec, n_enrich_vec, strd_vec, n_channel, bias_init=0.001):
    
    
    '''if write_summaries:
        tf.summary.image('input image', slice_and_scale(X))'''
    
    inp = X
    
    # Enc.
    
    for i in range(len(n_feat_vec)):
        
        if i==0:
            n_feat_in= n_channel
        else:
            n_feat_in= n_feat_vec[i-1]
        n_feat_out= n_feat_vec[i]
        
        ks= ks_0
        strd= 2
        
        n_l= n_feat_in*ks**3
        s_dev= np.sqrt(2.0/n_l)
        name_w= 'W_'+ str(i) + '_down'
        name_b= 'b_'+ str(i) + '_down'
        W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat_in,n_feat_out], stddev=s_dev), name=name_w)
        b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
        inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
            
        if n_enrich_vec[i]>0:
            
            inp0= inp
            
            n_feat_in= n_feat_out= n_feat_vec[i]
            ks= ks_0
            strd= 1
            
            for j in range(n_enrich_vec[i]):
                
                n_l= n_feat_in*ks**3
                s_dev= np.sqrt(2.0/n_l)
                name_w= 'W_'+ str(i) + '_' + str(j) + '_enrich_down'
                name_b= 'b_'+ str(i) + '_' + str(j) + '_enrich_down'
                W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat_in,n_feat_out], stddev=s_dev), name=name_w)
                b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
                inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
            
            inp+= inp0
            
        latent_vec= inp
        
    for i in range(len(n_feat_vec)-1,-1,-1):
        
        if i==0:
            n_feat_out= n_channel
        else:
            n_feat_out= n_feat_vec[i-1]
        n_feat_in= n_feat_vec[i]
        
        ks= ks_0
        strd= 2
        
        n_l= n_feat_in*ks**3
        s_dev= np.sqrt(2.0/n_l)
        name_w= 'W_'+ str(i) + '_up'
        name_b= 'b_'+ str(i) + '_up'
        W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat_out,n_feat_in], stddev=s_dev), name=name_w)
        b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
        in_shape = tf.shape(inp)
        out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]*2, n_feat_out])
        inp = tf.nn.relu(tf.add(tf.nn.conv3d_transpose(inp, W_1, out_shape, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
        
        if n_enrich_vec[i]>0:
            
            inp0= inp
            
            if i==0:
                n_feat_in= n_feat_out= n_channel
            else:
                n_feat_in= n_feat_out= n_feat_vec[i-1]
                
            ks= ks_0
            strd= 1
            
            for j in range(n_enrich_vec[i]):
                
                n_l= n_feat_in*ks**3
                s_dev= np.sqrt(2.0/n_l)
                name_w= 'W_'+ str(i) + '_' + str(j) + '_enrich_up'
                name_b= 'b_'+ str(i) + '_' + str(j) + '_enrich_up'
                W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat_in,n_feat_out], stddev=s_dev), name=name_w)
                b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_out]), name=name_b)
                inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
            
            inp+= inp0
            
    
    return inp, latent_vec








def cost_dice_forground(Y, predicter, loss_type= 'sorensen', smooth = 1e-5):
    
    target = Y[:, :, :, :, 1]
    output = predicter[:, :, :, :, 1]
    inse = tf.reduce_sum(output * target)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output)
        r = tf.reduce_sum(target * target)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output)
        r = tf.reduce_sum(target)
    dice = (2. * inse + smooth) / (l + r + smooth)
    #dice_whole = tf.reduce_mean(dice)

    cost= 1.0 - dice
    
    return cost



def cost_dice_multi_forground(Y, predicter, n_foreground=2, loss_type='sorensen', smooth=1e-5):

    cost= n_foreground

    for i_f in range(n_foreground):

        target = Y[:, :, :, :, i_f+1]
        output = predicter[:, :, :, :, i_f+1]
        inse = tf.reduce_sum(output * target)
        if loss_type == 'jaccard':
            l = tf.reduce_sum(output * output)
            r = tf.reduce_sum(target * target)
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(output)
            r = tf.reduce_sum(target)
        dice = (2. * inse + smooth) / (l + r + smooth)
        # dice_whole = tf.reduce_mean(dice)

        cost -= dice

    return cost



def cost_dice_full(Y, predicter, n_class= 2, loss_type='sorensen', smooth = 1e-5):
    
    cost = n_class
    for dice_channel in range(n_class):
        target = Y[:, :, :, :, dice_channel]
        output = predicter[:, :, :, :, dice_channel]
        inse = tf.reduce_sum(output * target, )
        if loss_type == 'jaccard':
            l = tf.reduce_sum(output * output)
            r = tf.reduce_sum(target * target)
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(output)
            r = tf.reduce_sum(target)
        dice = (2. * inse + smooth) / (l + r + smooth)
        dice_whole = tf.reduce_mean(dice, name='dice_coe')
        cost -= dice_whole
    
    return cost




def cost_focal(Y, predicter, gamma= 2.0):
    
    p_t = (predicter[:, :, :, :, 0] * Y[:, :, :, :, 0] + predicter[:, :, :, :, 1] * Y[:, :, :, :, 1] + 1e-5)
    p_t_2 = - tf.log(p_t) * tf.pow((1 - p_t), gamma)
    cost = tf.reduce_mean(p_t_2)
    
    return cost




def cost_x_entropy(Y, logit_f):
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_f, labels=Y))
    
    return cost





def cost_uncertainty_aleatoric(Y, logit_f, logit_s, EPS, T, loss_type='sorensen', smooth = 1e-5):
    
    cost= 0
    for iT in range(T):
        
       Eps= EPS[:,:,:,:,:,iT]
       #logit_f_n  = logit_f + Eps * tf.exp( logit_s )
       logit_f_n  = logit_f + Eps * tf.exp( tf.tile(logit_s, [1,1,1,1,2]) )
       logit_f_n_softmax= tf.nn.softmax(logit_f_n)
       
       target = Y[:, :, :, :, 1]
       output = logit_f_n_softmax[:, :, :, :, 1] * tf.exp( - logit_s[:, :, :, :, 0] )
       inse = tf.reduce_sum(output * target)
       if loss_type == 'jaccard':
           l = tf.reduce_sum(output * output)
           r = tf.reduce_sum(target * target)
       elif loss_type == 'sorensen':
           l = tf.reduce_sum(output)
           r = tf.reduce_sum(target)
       dice = (2. * inse + smooth) / (l + r + smooth)
       #dice_whole = tf.reduce_mean(dice)
       
       cost+= - dice + tf.reduce_mean( logit_s )
       
    return cost
  
    
    
def cost_uncertainty_epistemic(Y, predicter, logit_s, loss_type='sorensen', smooth = 1e-5):
    
    target = Y[:, :, :, :, 1]
    output = predicter[:, :, :, :, 1] * tf.exp( - logit_s[:, :, :, :, 0] )
    inse = tf.reduce_sum(output * target)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output)
        r = tf.reduce_sum(target * target)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output)
        r = tf.reduce_sum(target)
    dice = (2. * inse + smooth) / (l + r + smooth)
    #dice_whole = tf.reduce_mean(dice)
    
    cost= - dice + tf.reduce_mean( logit_s )
    
    return cost




def cost_tvesrky(Y, predicter, smooth = 1e-5):
    
    target = Y[:, :, :, :, 1]
    output = predicter[:, :, :, :, 1]
    inse = tf.reduce_sum(output * target, )
    diff1= tf.reduce_sum((1-output) * target)
    diff2= tf.reduce_sum(output * (1-target))
    cost = 1- inse/(inse+0.6*diff1+0.4*diff2+smooth)
    
    return cost




def cost_mae(Y, predicter):
    
    target = Y[:, :, :, :, :]
    output = predicter[:, :, :, :, :]
    cost = tf.reduce_mean( tf.abs( output - target ) )
    
    return cost



def cost_lq(Y, predicter, q=0.7):
    
    target = Y[:, :, :, :, 1]
    output = predicter[:, :, :, :, 1]
    cost = tf.reduce_sum( tf.multiply(target, 1- tf.pow( tf.multiply(target,output) , q ) ) / q )
    
    return cost




def davood_net_multiple(X, ks_0, depth, n_feat_0, n_channel, n_class, p_keep_conv, bias_init=0.001, write_summaries= False, model_no= 1):
    
    model_str= str(model_no)
    
    feat_fine = [None] * (depth - 1)
    
    if write_summaries:
        tf.summary.image('input image', slice_and_scale(X))
    
    for level in range(depth):
        
        ks = ks_0
                
        if level == 0:
            
            strd = 1
            
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init' + model_str
            name_b = 'b_' + str(level) + '_init' + model_str
            name_c = 'Conv_' + str(level) + '_init' + model_str
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
        else:
            
            strd = 2
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init' + model_str
            name_b = 'b_' + str(level) + '_init' + model_str
            name_c = 'Conv_' + str(level) + '_init' + model_str
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            for i in range(1, level):
                n_l = n_feat_0 * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                name_w = 'W_' + str(level) + '_' + str(i) + '_init' + model_str
                name_b = 'b_' + str(level) + '_' + str(i) + '_init' + model_str
                name_c = 'Conv_' + str(level) + '_' + str(i) + '_init' + model_str
                W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat_0, n_feat_0], stddev=s_dev), name=name_w)
                b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
                inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                inp= tf.nn.dropout(inp, p_keep_conv)
                
            for level_reg in range(0, level):
                
                inp_0 = feat_fine[level_reg]
                
                level_diff = level - level_reg
                
                n_feat = n_feat_0 * 2 ** level_reg
                n_l = n_feat * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                
                for j in range(level_diff):
                    name_w = 'W_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg' + model_str
                    name_b = 'b_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg' + model_str
                    name_c = 'Conv_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg' + model_str
                    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
                    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
                    inp_0 = tf.nn.relu(
                        tf.add(tf.nn.conv3d(inp_0, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                    inp_0 = tf.nn.dropout(inp_0, p_keep_conv)
                    
                inp = tf.concat([inp, inp_0], 4)
                
        ks = ks_0
        
        n_feat = n_feat_0 * 2 ** level
        
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_down' + model_str
            name_b = 'b_' + str(level) + '_2_down' + model_str
            name_c = 'Conv_' + str(level) + '_2_down' + model_str
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_down' + model_str
            name_b = 'b_' + str(level) + '_3_down' + model_str
            name_c = 'Conv_' + str(level) + '_3_down' + model_str
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_down' + model_str
            name_b = 'b_' + str(level) + '_4_down' + model_str
            name_c = 'Conv_' + str(level) + '_4_down' + model_str
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_down' + model_str
            name_b = 'b_' + str(level) + '_5_down' + model_str
            name_c = 'Conv_' + str(level) + '_5_down' + model_str
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_1 + inp_0  ###
            
        if level < depth - 1:
            feat_fine[level] = inp
            
    # DeConvolution Layers
    
    for level in range(depth - 2, -1, -1):
        
        ks = ks_0
        
        n_l = n_feat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_up' + model_str
        name_b = 'b_' + str(level) + '_up' + model_str
        name_c = 'Conv_' + str(level) + '_up' + model_str
        W_deconv = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat // 2, n_feat], stddev=s_dev), name=name_w)
        b_deconv = tf.Variable(tf.constant(bias_init, shape=[n_feat // 2]), name=name_b)
        in_shape = tf.shape(inp)
        if level == 3:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, 9, in_shape[4] // 2])
        else:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, in_shape[4] // 2])
        Deconv = tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
        Deconv = tf.nn.relu(tf.add(Deconv, b_deconv))
        Deconv= tf.nn.dropout(Deconv, p_keep_conv)
        inp = tf.concat([feat_fine[level], Deconv], 4)
        
        if write_summaries:
            tf.summary.histogram(name_w, W_deconv)
            tf.summary.histogram(name_b, b_deconv)
            tf.summary.image(name_c, slice_and_scale(inp))
            
        if level == depth - 2:
            n_concat = n_feat
        else:
            n_concat = n_feat * 3 // 4
            
        if level < depth - 2:
            n_feat = n_feat // 2
            
        n_l = n_concat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_1_up' + model_str
        name_b = 'b_' + str(level) + '_1_up' + model_str
        name_c = 'Conv_' + str(level) + '_1_up' + model_str
        W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_concat, n_feat], stddev=s_dev), name=name_w)
        b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
        inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1))
        inp= tf.nn.dropout(inp, p_keep_conv)
        
        if write_summaries:
            tf.summary.histogram(name_w, W_1)
            tf.summary.histogram(name_b, b_1)
            tf.summary.image(name_c, slice_and_scale(inp))
            
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_up' + model_str
            name_b = 'b_' + str(level) + '_2_up' + model_str
            name_c = 'Conv_' + str(level) + '_2_up' + model_str
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_up' + model_str
            name_b = 'b_' + str(level) + '_3_up' + model_str
            name_c = 'Conv_' + str(level) + '_3_up' + model_str
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_up' + model_str
            name_b = 'b_' + str(level) + '_4_up' + model_str
            name_c = 'Conv_' + str(level) + '_4_up' + model_str
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_up' + model_str
            name_b = 'b_' + str(level) + '_5_up' + model_str
            name_c = 'Conv_' + str(level) + '_5_up' + model_str
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_1 + inp_0  ###
            
     #        if level==0:
     #            n_l= n_feat*ks**3
     #            s_dev= np.sqrt(2.0/n_l)
     #            name_w= 'W_up'
     #            name_b= 'b_up'
     #            name_c= 'Conv_up'
     #            W_deconv= tf.Variable(tf.truncated_normal([ks,ks,ks,n_class,n_feat], stddev=s_dev), name=name_w)
     #            b_deconv= tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
     #            in_shape = tf.shape(inp)
     #            out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]*2, n_class])
     #            Deconv= tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1,2,2,2,1], padding='SAME')
     #            output= tf.add(Deconv, b_deconv)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w = 'W_out' + model_str
    name_b = 'b_out' + model_str
#    name_c = 'Conv_out'
    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_class], stddev=s_dev), name=name_w)
    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
    output = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1)
    # output= tf.nn.dropout(output, p_keep_conv)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w_s = 'W_out_s' + model_str
    name_b_s = 'b_out_s' + model_str
#    name_c_s = 'Conv_out_s'
    W_1_s = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, 1], stddev=s_dev), name=name_w_s)
    b_1_s = tf.Variable(tf.constant(bias_init, shape=[1]), name=name_b_s)
    output_s = tf.add(tf.nn.conv3d(inp, W_1_s, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1_s)
    # output= tf.nn.dropout(output, p_keep_conv)
    
    '''if write_summaries:
        tf.summary.histogram(name_w, W_1)
        tf.summary.histogram(name_b, b_1)
        tf.summary.image(name_c, slice_and_scale(inp))'''

    return output, output_s






def davood_reg_net(X, n_feat_vec, p_keep_hidden, bias_init=0.001):
    
    inp = X
    
    for i in range(len(n_feat_vec)-1):
        
        n_feat_in= n_feat_vec[i]
        n_feat_out= n_feat_vec[i+1]
        
        W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_fc_'+str(i))
        b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc_'+str(i))
        inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
        
        inp = tf.nn.dropout(inp, p_keep_hidden)
    
    return inp














def davood_net_return_fmaps(X, ks_0, depth, n_feat_0, n_channel, n_class, p_keep_conv, bias_init=0.001, write_summaries= False):
    
    feat_fine = [None] * (depth - 1)
    
    if write_summaries:
        tf.summary.image('input image', slice_and_scale(X))
    
    for level in range(depth):
        
        ks = ks_0
                
        if level == 0:
            
            strd = 1
            
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            name_c = 'Conv_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
        else:
            
            strd = 2
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            name_c = 'Conv_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            for i in range(1, level):
                n_l = n_feat_0 * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                name_w = 'W_' + str(level) + '_' + str(i) + '_init'
                name_b = 'b_' + str(level) + '_' + str(i) + '_init'
                name_c = 'Conv_' + str(level) + '_' + str(i) + '_init'
                W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat_0, n_feat_0], stddev=s_dev), name=name_w)
                b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
                inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                inp= tf.nn.dropout(inp, p_keep_conv)
                
            for level_reg in range(0, level):
                
                inp_0 = feat_fine[level_reg]
                
                level_diff = level - level_reg
                
                n_feat = n_feat_0 * 2 ** level_reg
                n_l = n_feat * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                
                for j in range(level_diff):
                    name_w = 'W_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    name_b = 'b_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    name_c = 'Conv_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
                    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
                    inp_0 = tf.nn.relu(
                        tf.add(tf.nn.conv3d(inp_0, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                    inp_0 = tf.nn.dropout(inp_0, p_keep_conv)
                    
                inp = tf.concat([inp, inp_0], 4)
                
        ks = ks_0
        
        n_feat = n_feat_0 * 2 ** level
        
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_down'
            name_b = 'b_' + str(level) + '_2_down'
            name_c = 'Conv_' + str(level) + '_2_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_down'
            name_b = 'b_' + str(level) + '_3_down'
            name_c = 'Conv_' + str(level) + '_3_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_down'
            name_b = 'b_' + str(level) + '_4_down'
            name_c = 'Conv_' + str(level) + '_4_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_down'
            name_b = 'b_' + str(level) + '_5_down'
            name_c = 'Conv_' + str(level) + '_5_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_1 + inp_0  ###
            
        if level < depth - 1:
            feat_fine[level] = inp
            
    # DeConvolution Layers
    
    feat_cors = [None] * (depth - 1)
    
    for level in range(depth - 2, -1, -1):
        
        ks = ks_0
        
        n_l = n_feat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_up'
        name_b = 'b_' + str(level) + '_up'
        name_c = 'Conv_' + str(level) + '_up'
        W_deconv = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat // 2, n_feat], stddev=s_dev), name=name_w)
        b_deconv = tf.Variable(tf.constant(bias_init, shape=[n_feat // 2]), name=name_b)
        in_shape = tf.shape(inp)
        if level == 3:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, 9, in_shape[4] // 2])
        else:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, in_shape[4] // 2])
        Deconv = tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
        Deconv = tf.nn.relu(tf.add(Deconv, b_deconv))
        Deconv= tf.nn.dropout(Deconv, p_keep_conv)
        inp = tf.concat([feat_fine[level], Deconv], 4)
        
        if write_summaries:
            tf.summary.histogram(name_w, W_deconv)
            tf.summary.histogram(name_b, b_deconv)
            tf.summary.image(name_c, slice_and_scale(inp))
            
        if level == depth - 2:
            n_concat = n_feat
        else:
            n_concat = n_feat * 3 // 4
            
        if level < depth - 2:
            n_feat = n_feat // 2
            
        n_l = n_concat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_1_up'
        name_b = 'b_' + str(level) + '_1_up'
        name_c = 'Conv_' + str(level) + '_1_up'
        W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_concat, n_feat], stddev=s_dev), name=name_w)
        b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
        inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1))
        inp= tf.nn.dropout(inp, p_keep_conv)
        
        if write_summaries:
            tf.summary.histogram(name_w, W_1)
            tf.summary.histogram(name_b, b_1)
            tf.summary.image(name_c, slice_and_scale(inp))
            
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_up'
            name_b = 'b_' + str(level) + '_2_up'
            name_c = 'Conv_' + str(level) + '_2_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_up'
            name_b = 'b_' + str(level) + '_3_up'
            name_c = 'Conv_' + str(level) + '_3_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_up'
            name_b = 'b_' + str(level) + '_4_up'
            name_c = 'Conv_' + str(level) + '_4_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_up'
            name_b = 'b_' + str(level) + '_5_up'
            name_c = 'Conv_' + str(level) + '_5_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_1 + inp_0  ###
        
        feat_cors[level] = inp
        
     #        if level==0:
     #            n_l= n_feat*ks**3
     #            s_dev= np.sqrt(2.0/n_l)
     #            name_w= 'W_up'
     #            name_b= 'b_up'
     #            name_c= 'Conv_up'
     #            W_deconv= tf.Variable(tf.truncated_normal([ks,ks,ks,n_class,n_feat], stddev=s_dev), name=name_w)
     #            b_deconv= tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
     #            in_shape = tf.shape(inp)
     #            out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]*2, n_class])
     #            Deconv= tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1,2,2,2,1], padding='SAME')
     #            output= tf.add(Deconv, b_deconv)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w = 'W_out'
    name_b = 'b_out'
#    name_c = 'Conv_out'
    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_class], stddev=s_dev), name=name_w)
    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
    output = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1)
    # output= tf.nn.dropout(output, p_keep_conv)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w_s = 'W_out_s'
    name_b_s = 'b_out_s'
#    name_c_s = 'Conv_out_s'
    W_1_s = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, 1], stddev=s_dev), name=name_w_s)
    b_1_s = tf.Variable(tf.constant(bias_init, shape=[1]), name=name_b_s)
    output_s = tf.add(tf.nn.conv3d(inp, W_1_s, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1_s)
    # output= tf.nn.dropout(output, p_keep_conv)
    
    '''if write_summaries:
        tf.summary.histogram(name_w, W_1)
        tf.summary.histogram(name_b, b_1)
        tf.summary.image(name_c, slice_and_scale(inp))'''

    return output, output_s, feat_fine, feat_cors






























def davood_net_frcnn(X, ks_0, depth, n_feat_0, n_channel, n_class, n_anchor, p_keep_conv, bias_init=0.001, write_summaries= False):
    
    feat_fine = [None] * (depth - 1)
    
    if write_summaries:
        tf.summary.image('input image', slice_and_scale(X))
    
    for level in range(depth):
        
        ks = ks_0
                
        if level == 0:
            
            strd = 1
            
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            name_c = 'Conv_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
        else:
            
            strd = 2
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            name_c = 'Conv_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            for i in range(1, level):
                n_l = n_feat_0 * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                name_w = 'W_' + str(level) + '_' + str(i) + '_init'
                name_b = 'b_' + str(level) + '_' + str(i) + '_init'
                name_c = 'Conv_' + str(level) + '_' + str(i) + '_init'
                W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat_0, n_feat_0], stddev=s_dev), name=name_w)
                b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
                inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                inp= tf.nn.dropout(inp, p_keep_conv)
                
            for level_reg in range(0, level):
                
                inp_0 = feat_fine[level_reg]
                
                level_diff = level - level_reg
                
                n_feat = n_feat_0 * 2 ** level_reg
                n_l = n_feat * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                
                for j in range(level_diff):
                    name_w = 'W_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    name_b = 'b_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    name_c = 'Conv_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
                    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
                    inp_0 = tf.nn.relu(
                        tf.add(tf.nn.conv3d(inp_0, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                    inp_0 = tf.nn.dropout(inp_0, p_keep_conv)
                    
                inp = tf.concat([inp, inp_0], 4)
                
        ks = ks_0
        
        n_feat = n_feat_0 * 2 ** level
        
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_down'
            name_b = 'b_' + str(level) + '_2_down'
            name_c = 'Conv_' + str(level) + '_2_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_down'
            name_b = 'b_' + str(level) + '_3_down'
            name_c = 'Conv_' + str(level) + '_3_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_down'
            name_b = 'b_' + str(level) + '_4_down'
            name_c = 'Conv_' + str(level) + '_4_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_down'
            name_b = 'b_' + str(level) + '_5_down'
            name_c = 'Conv_' + str(level) + '_5_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_1 + inp_0  ###
            
        if level < depth - 1:
            feat_fine[level] = inp
            
    # DeConvolution Layers
    
    for level in range(depth - 2, -1, -1):
        
        ks = ks_0
        
        n_l = n_feat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_up'
        name_b = 'b_' + str(level) + '_up'
        name_c = 'Conv_' + str(level) + '_up'
        W_deconv = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat // 2, n_feat], stddev=s_dev), name=name_w)
        b_deconv = tf.Variable(tf.constant(bias_init, shape=[n_feat // 2]), name=name_b)
        in_shape = tf.shape(inp)
        if level == 3:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, 9, in_shape[4] // 2])
        else:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, in_shape[4] // 2])
        Deconv = tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
        Deconv = tf.nn.relu(tf.add(Deconv, b_deconv))
        Deconv= tf.nn.dropout(Deconv, p_keep_conv)
        inp = tf.concat([feat_fine[level], Deconv], 4)
        
        if write_summaries:
            tf.summary.histogram(name_w, W_deconv)
            tf.summary.histogram(name_b, b_deconv)
            tf.summary.image(name_c, slice_and_scale(inp))
            
        if level == depth - 2:
            n_concat = n_feat
        else:
            n_concat = n_feat * 3 // 4
            
        if level < depth - 2:
            n_feat = n_feat // 2
            
        n_l = n_concat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_1_up'
        name_b = 'b_' + str(level) + '_1_up'
        name_c = 'Conv_' + str(level) + '_1_up'
        W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_concat, n_feat], stddev=s_dev), name=name_w)
        b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
        inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1))
        inp= tf.nn.dropout(inp, p_keep_conv)
        
        if write_summaries:
            tf.summary.histogram(name_w, W_1)
            tf.summary.histogram(name_b, b_1)
            tf.summary.image(name_c, slice_and_scale(inp))
            
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_up'
            name_b = 'b_' + str(level) + '_2_up'
            name_c = 'Conv_' + str(level) + '_2_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_up'
            name_b = 'b_' + str(level) + '_3_up'
            name_c = 'Conv_' + str(level) + '_3_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_up'
            name_b = 'b_' + str(level) + '_4_up'
            name_c = 'Conv_' + str(level) + '_4_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_up'
            name_b = 'b_' + str(level) + '_5_up'
            name_c = 'Conv_' + str(level) + '_5_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_1 + inp_0  ###
            
    
    ks= 1
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w = 'W_prob'
    name_b = 'b_prob'
    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_class*n_anchor], stddev=s_dev), name=name_w)
    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_class*n_anchor]), name=name_b)
    prob = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w = 'W_reg'
    name_b = 'b_reg'
    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, 6*n_anchor], stddev=s_dev), name=name_w)
    b_1 = tf.Variable(tf.constant(bias_init, shape=[6*n_anchor]), name=name_b)
    reg = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1)
    
        
    return prob, reg





















def davood_net_r(X, ks_0, depth, n_feat_0, n_channel, n_class, p_keep_conv, bias_init=0.001, write_summaries= False):
    
    feat_fine = [None] * (depth - 1)
    
    if write_summaries:
        tf.summary.image('input image', slice_and_scale(X))
    
    for level in range(depth):
        
        ks = ks_0
                
        if level == 0:
            
            strd = 1
            
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            name_c = 'Conv_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
        else:
            
            strd = 2
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            name_c = 'Conv_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            for i in range(1, level):
                n_l = n_feat_0 * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                name_w = 'W_' + str(level) + '_' + str(i) + '_init'
                name_b = 'b_' + str(level) + '_' + str(i) + '_init'
                name_c = 'Conv_' + str(level) + '_' + str(i) + '_init'
                W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat_0, n_feat_0], stddev=s_dev), name=name_w)
                b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
                inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                inp= tf.nn.dropout(inp, p_keep_conv)
                
            for level_reg in range(0, level):
                
                inp_0 = feat_fine[level_reg]
                
                level_diff = level - level_reg
                
                n_feat = n_feat_0 * 2 ** level_reg
                n_l = n_feat * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                
                for j in range(level_diff):
                    name_w = 'W_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    name_b = 'b_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    name_c = 'Conv_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
                    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
                    inp_0 = tf.nn.relu(
                        tf.add(tf.nn.conv3d(inp_0, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                    inp_0 = tf.nn.dropout(inp_0, p_keep_conv)
                    
                inp = tf.concat([inp, inp_0], 4)
                
        ks = ks_0
        
        n_feat = n_feat_0 * 2 ** level
        
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_down'
            name_b = 'b_' + str(level) + '_2_down'
            name_c = 'Conv_' + str(level) + '_2_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_down'
            name_b = 'b_' + str(level) + '_3_down'
            name_c = 'Conv_' + str(level) + '_3_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_down'
            name_b = 'b_' + str(level) + '_4_down'
            name_c = 'Conv_' + str(level) + '_4_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_down'
            name_b = 'b_' + str(level) + '_5_down'
            name_c = 'Conv_' + str(level) + '_5_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_1 + inp_0  ###
            
        if level < depth - 1:
            feat_fine[level] = inp
            
    # DeConvolution Layers
    
    for level in range(depth - 2, -1, -1):
        
        ks = ks_0
        
        n_l = n_feat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_up'
        name_b = 'b_' + str(level) + '_up'
        name_c = 'Conv_' + str(level) + '_up'
        W_deconv = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat // 2, n_feat], stddev=s_dev), name=name_w)
        b_deconv = tf.Variable(tf.constant(bias_init, shape=[n_feat // 2]), name=name_b)
        in_shape = tf.shape(inp)
        if level == 3:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, 9, in_shape[4] // 2])
        else:
            out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, in_shape[4] // 2])
        Deconv = tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
        Deconv = tf.nn.relu(tf.add(Deconv, b_deconv))
        Deconv= tf.nn.dropout(Deconv, p_keep_conv)
        inp = tf.concat([feat_fine[level], Deconv], 4)
        
        if write_summaries:
            tf.summary.histogram(name_w, W_deconv)
            tf.summary.histogram(name_b, b_deconv)
            tf.summary.image(name_c, slice_and_scale(inp))
            
        if level == depth - 2:
            n_concat = n_feat
        else:
            n_concat = n_feat * 3 // 4
            
        if level < depth - 2:
            n_feat = n_feat // 2
            
        n_l = n_concat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_1_up'
        name_b = 'b_' + str(level) + '_1_up'
        name_c = 'Conv_' + str(level) + '_1_up'
        W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_concat, n_feat], stddev=s_dev), name=name_w)
        b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
        inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1))
        inp= tf.nn.dropout(inp, p_keep_conv)
        
        if write_summaries:
            tf.summary.histogram(name_w, W_1)
            tf.summary.histogram(name_b, b_1)
            tf.summary.image(name_c, slice_and_scale(inp))
            
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_up'
            name_b = 'b_' + str(level) + '_2_up'
            name_c = 'Conv_' + str(level) + '_2_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_up'
            name_b = 'b_' + str(level) + '_3_up'
            name_c = 'Conv_' + str(level) + '_3_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_up'
            name_b = 'b_' + str(level) + '_4_up'
            name_c = 'Conv_' + str(level) + '_4_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_up'
            name_b = 'b_' + str(level) + '_5_up'
            name_c = 'Conv_' + str(level) + '_5_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
                
            inp = inp + inp_1 + inp_0  ###
            
     #        if level==0:
     #            n_l= n_feat*ks**3
     #            s_dev= np.sqrt(2.0/n_l)
     #            name_w= 'W_up'
     #            name_b= 'b_up'
     #            name_c= 'Conv_up'
     #            W_deconv= tf.Variable(tf.truncated_normal([ks,ks,ks,n_class,n_feat], stddev=s_dev), name=name_w)
     #            b_deconv= tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
     #            in_shape = tf.shape(inp)
     #            out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]*2, n_class])
     #            Deconv= tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1,2,2,2,1], padding='SAME')
     #            output= tf.add(Deconv, b_deconv)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w = 'W_out'
    name_b = 'b_out'
#    name_c = 'Conv_out'
    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_class], stddev=s_dev), name=name_w)
    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
    output = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1)
    # output= tf.nn.dropout(output, p_keep_conv)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w_s = 'W_out_s'
    name_b_s = 'b_out_s'
#    name_c_s = 'Conv_out_s'
    W_1_s = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_channel], stddev=s_dev), name=name_w_s)
    b_1_s = tf.Variable(tf.constant(bias_init, shape=[n_channel]), name=name_b_s)
    output_r = tf.add(tf.nn.conv3d(inp, W_1_s, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1_s)
    # output= tf.nn.dropout(output, p_keep_conv)
    
    '''if write_summaries:
        tf.summary.histogram(name_w, W_1)
        tf.summary.histogram(name_b, b_1)
        tf.summary.image(name_c, slice_and_scale(inp))'''
        
    return output, output_r



















    



