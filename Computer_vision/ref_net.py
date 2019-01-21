# -*- coding: utf-8 -*-
"""
Spyder Editor

Scripts for designing RefNet, with data augmentations functions
"""


# importing libraries
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
import skimage
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
import cv2

from keras.models import Model, load_model

from keras.layers import Input, Activation, ZeroPadding2D, Add
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import glorot_uniform
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

TRAIN_PATH = 'dataset/stage1_train'
TEST_PATH = 'dataset/stage1_test'

seed = 17
random.seed = seed
np.random.seed = seed



def get_data():
    # Get train and test IDs
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]
    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + "/" + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask
    # Get and resize test images
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + '/' + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
    print('Done!')
    return X_train, Y_train, X_test

def u_net_model(img_shape, verbose=1):
    inputs = Input(img_shape)
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    if verbose == 1:
        model.summary()

    return model
    

"""
    Define the functions for encoding operations in Refine-Net
"""
def ResNetBlock(X_prev, filters):
    """
        X_prev -- the weights layers from the previous layer with the shape(n_h, n_w, n_c)
        filters -- the python integer of channels number(it shoud be equal to n_c)
        
        returns:
        X -- weights layer with the shape(n_h, n_w, n_c)
    """    
#     (n_h, n_w) = X_prev.shape[:2]
#     X_bat = Input((n_h, n_w, filters))
    X_prev = Conv2D(filters, kernel_size=(1,1), strides=(1,1), kernel_initializer='he_normal')(X_prev)
    X_shortcut = X_prev
    
    
    X = BatchNormalization(axis=3)(X_prev)
    X = Activation('relu')(X)
    X = Conv2D(filters, kernel_size=(3,3), strides=(1,1), kernel_initializer='he_normal', padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters, kernel_size=(3,3), strides=(1,1), kernel_initializer='he_normal', padding='same')(X)
    
    # shortcut connection
    X = Add()([X, X_shortcut])
    
    return X

def RCU(X_prev, filters):
    """
            Residual Conv Unit 
        ReLU -> Conv 3x3 -> ReLU -> Conv 3x3 -> identity function(X, X_prev)
        
        X_prev -- weight from the corresponding unit of UNet
        filters -- the python integer of channels number, better to take the quantity from the prev layer
        
        
        UPD: ADDING BOTTLENECK
    """
    X_prev = Conv2D(filters, kernel_size=(1,1), strides=(1,1), kernel_initializer='he_normal')(X_prev)
    X_shortcut = X_prev
    
    X = Activation('relu')(X_prev)
    X = Conv2D(np.int32(filters/4), (1,1), kernel_initializer='he_normal')(X)
    X = Conv2D(filters, kernel_size=(3,3), strides=(1,1), kernel_initializer='he_normal', padding='same')(X)
    
    X = Activation('relu')(X)
    
    X = Conv2D(np.int32(filters/4), (1,1), kernel_initializer='he_normal')(X)
    X = Conv2D(filters, kernel_size=(3,3), strides=(1,1), kernel_initializer='he_normal', padding='same')(X)
    
    X = Add()([X, X_shortcut])
    
    return X

def multi_resolution(X_prev, filters, upsample=True, up=2):
    """
        Multi-resolution Unit
            if the input is unitary - skip the unit
        Conv 3x3 -> Upsampling by the factor of 2 if the input is of different shapes -> Sum the results to one matrix
        Parameters:
        X_prev -- output from RCU
        filters -- n_c from the prev channel
        up -- the size we need input image to mulitply
        upsample -- boolean which define the need to upsample the image to the size of the biggest one
    """
    if upsample == True:
        X = Conv2D(np.int32(filters/4), (1,1), kernel_initializer='he_normal')(X_prev)
        X = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal')(X)
        X = Conv2DTranspose(filters, (up,up), strides=(up,up), padding = 'same')(X)
    else:
        X = Conv2D(np.int32(filters/4), (1,1), kernel_initializer='he_normal')(X_prev)
        X = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal')(X)    
    
    return X
        
        
def add_multi_resolution(X1, X2):
    """
        Function for adding two tensors for the purpose of Multi-resolution Unit
    """
    X = Add()([X1, X2])
    return X

def chain_pool(X_prev, filters):
    """
        Chained residual pooling unit
           ReLU(X_prev) -> (5x5 MaxPool -> 3x3 Conv)x2 == X -> Sum(ReLU(X_prev), X)
           
    """
    X_prev = Activation('relu')(X_prev)
    X = MaxPooling2D((5,5), strides=(1,1), padding = 'same')(X_prev)
    X = Conv2D(np.int32(filters/4), (1,1), kernel_initializer='he_normal')(X)
    X = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal')(X)
    
#     X = MaxPooling2D((5,5), strides=(1,1), padding = 'same')(X)
#     X = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal')(X)
    
    X = Add()([X, X_prev])
    return X

def RefNet(input_shape=(256,256,3), verbose=1):
    
    # SOME CONSTANTS
    F = 256   # Filters for each layer
    
    X_input = Input(input_shape)
    s = Lambda(lambda x: x / 255) (X_input)
    
    ################# STAGE OF ENCODING BY UNET
    # Stage 1
    c1_1 = Conv2D(8, (3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation='relu')(s)
    c1_2 = Conv2D(8, (3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation='relu')(c1_1)
    m1 = MaxPooling2D((2,2))(c1_2)
    
    # Stage 2-6 ResNet blocks
    rs_2 = ResNetBlock(m1, 16)
    m2 = MaxPooling2D((2,2))(rs_2)
    
    rs_3 = ResNetBlock(m2, 32)
    m3 = MaxPooling2D((2,2))(rs_3)
    
    rs_4 = ResNetBlock(m3, 64)
    m4 = MaxPooling2D((2,2))(rs_4)
    
    rs_5 = ResNetBlock(m4, 128)
    m5 = MaxPooling2D((2,2))(rs_5)
    
    rs_6 = ResNetBlock(m5, 256)
    
    ################# END STAGE OF ENCODING, GO TO REFNET
    
    ### REFNET-4 INPUT 1/32
    # ADAPTIVE CONV FOR RS_6 OR 1/32 OF IMAGE, UNITARY INPUT
    # 512 filters for refnet-4 for else -- 256
    rcu_6_1 = RCU(rs_6, F)
    # rcu_6_2 = RCU(rcu_6_1, F4)
    
    # SKIP MULTI RESOLUTION UNIT #
    
    # CHAIN POOLING
    ch_res_pool_6 = chain_pool(rcu_6_1, F)

    # OUTPUT CONV RCU
    out_conv_6 = RCU(ch_res_pool_6, F)
    
    #### END REFNET-4
    
    ### REFNET-3 MULTIPLE INPUT: 1/32(out_conv_6), 1/16(rs_5)
    
        # ADAPTIVE CONV RCU
    rcu_5_1_sm = RCU(out_conv_6, F)
    
    rcu_5_1_bg = RCU(rs_5, F)
    
        # MULTIRESOLUTION UNIT
    ml_res_5_sm = multi_resolution(rcu_5_1_sm, F, upsample=True)
    ml_res_5_bg = multi_resolution(rcu_5_1_bg, F, upsample=False)
    
        # ADDING ML OUTPUTS
    ml_res_5_add = add_multi_resolution(ml_res_5_sm, ml_res_5_bg)
    
        # CHAIN POOLING
    ch_res_pool_5 = chain_pool(ml_res_5_add, F)
    
        # OUTPUT CONV RCU
    out_conv_5 = RCU(ch_res_pool_5, F)
    
    #### END REFINE-NET-3 OUTPUT SHAPE: 16X16X256
    
    ### REFNET-2 MULTIPLE INPUT: 1/16(out_conv_5), 1/8(rs_4)
        # ADAPTIVE CONV RCU
    rcu_4_1_sm = RCU(out_conv_5, F)
    
    rcu_4_1_bg = RCU(rs_4, F)
    
        # MULTIRESOLUTION UNIT
    ml_res_4_sm = multi_resolution(rcu_4_1_sm, F, upsample=True)
    ml_res_4_bg = multi_resolution(rcu_4_1_bg, F, upsample=False)
    
        # ADDING ML OUTPUTS
    ml_res_4_add = add_multi_resolution(ml_res_4_sm, ml_res_4_bg)
    
        # CHAIN POOLING
    ch_res_pool_4 = chain_pool(ml_res_4_add, F)
    
        # OUTPUT CONV RCU
    out_conv_4 = RCU(ch_res_pool_4, F)
    
    ### END REFNET-2 OUT SHAPE: 32X32X256
    
    ### REFNET-1 MULTIPLE INPUT: 1/8(out_conv_4), 1/4(rs_3)
        # ADAPTIVE CONV RCU
    rcu_3_1_sm = RCU(out_conv_4, F)
    
    rcu_3_1_bg = RCU(rs_3, F)

    
        # MULTIRESOLUTION UNIT
    ml_res_3_sm = multi_resolution(rcu_3_1_sm, F, upsample=True)
    ml_res_3_bg = multi_resolution(rcu_3_1_bg, F, upsample=False)
    
        # ADDING ML OUTPUTS
    ml_res_3_add = add_multi_resolution(ml_res_3_sm, ml_res_3_bg)
    
        # CHAIN POOLING
    ch_res_pool_3 = chain_pool(ml_res_3_add, F)
    
        # OUTPUT CONV RCU
    out_conv_3 = RCU(ch_res_pool_3, F)
    ### END REFNET-2 OUT SHAPE: 64 X 64 X 256
    
    # UPSAMPLING THE OUT CONV TO THE ORIG SHAPE OF INPUT 
    up_to_orig = Conv2DTranspose(64, kernel_size=(4,4), strides=(4,4), padding='same', 
                                 kernel_initializer='he_normal')(out_conv_3)
    
    outputs = Conv2D(1, (1,1), activation='sigmoid')(up_to_orig)
    
    model = Model(inputs=[X_input], outputs=[outputs])
    
    if verbose==1:
        model.summary()
    return model

def SingleRefNet(input_shape=(256,256,3), verbose=1):
    """
        Single block Refine Net with residual blocks encoding
    
    """
    # SOME CONSTANTS
    F = 64   # FOR ELSE
    
    X_input = Input(input_shape)
    s = Lambda(lambda x: x / 255) (X_input)
    
    ################# STAGE OF ENCODING BY UNET
    # Stage 1
    c1_1 = Conv2D(16, (3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation='relu')(s)
    c1_2 = Conv2D(16, (3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation='relu')(c1_1)
    m1 = MaxPooling2D((2,2))(c1_2)
    
    # Stage 2-6 ResNet blocks
    rs_2 = ResNetBlock(m1, 32)
    m2 = MaxPooling2D((2,2))(rs_2)
    
    rs_3 = ResNetBlock(m2, 64)
    m3 = MaxPooling2D((2,2))(rs_3)
    
    rs_4 = ResNetBlock(m3, 128)
    
    ################# END STAGE OF ENCODING, GO TO REFNET
    
    ### SINGLE REFBLOCK 
    ### UPD: REDUCING THE QUANTITY OF FILTERS TO F=64
    ### INPUT: c1_2 | shape: (256,256,16)
    ###        rs_2 | shape: (128,128,32)
    ###        rs_3 | shape: (64,64,64)
    ###        rs_4 | shape: (32,32,128)
    
    ### OUTPUT: pred | shape: (256,256,64)
    
    ###############
    
    # RESIDUAL CONV UNIT 2X
    rcu_5_1 = RCU(rs_4, F)
    rcu_5_2 = RCU(rcu_5_1, F)
    
    rcu_5_3 = RCU(rs_3, F)
    rcu_5_4 = RCU(rcu_5_3, F)
    
    rcu_5_5 = RCU(rs_2, F)
    rcu_5_6 = RCU(rcu_5_5, F)
    
    rcu_5_7 = RCU(c1_2, F)
    
    ###############
    
    # MULTI-RESOLUTION UNIT
    # INPUT: rcu_5_2 | shape: (32,32,64)
    #        rcu_5_4 | shape: (64,64,64)
    #        rcu_5_6 | shape: (128,128,64)
    #        rcu_5_7 | shape: (256,256,64)
    # OUTPUT: ml_out | shape: (256,256,64)
    
    ml_res_1 = multi_resolution(rcu_5_2, F, upsample=True, up=8)
    ml_res_2 = multi_resolution(rcu_5_4, F, upsample=True, up=4)
    ml_res_3 = multi_resolution(rcu_5_6, F, upsample=True, up=2)
    ml_res_4 = multi_resolution(rcu_5_7, F, upsample=False)
    
    ## ADDING
    X = add_multi_resolution(ml_res_1, ml_res_2)
    Y = add_multi_resolution(ml_res_3, ml_res_4)
    ml_out = add_multi_resolution(X, Y)
    
    ### CHAIN RESOLUTION UNIT
    ch_p = chain_pool(ml_out, F)
    
    ### RCU
    out = RCU(ch_p, F)
    
    
    ### MAKING THE PREDICTIONS

    outputs = Conv2D(1, (1,1), activation='sigmoid')(out)
    
    model = Model(inputs=[X_input], outputs=[outputs])
    
    if verbose==1:
        model.summary()
    return model


def mean_iou(y_true, y_pred):
    """
        Mean Intersection of union. Metric for the competition
    """
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

####### FUNCTIONS FOR SUBMISSION ENCODING

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
###############


## DATA AUGMENTATION  ##
def transform_img(img, angle):
    M = cv2.getRotationMatrix2D((IMG_HEIGHT/2,IMG_HEIGHT/2),angle,1)
    transformed_img = cv2.warpAffine(np.uint8(img),M,(IMG_HEIGHT,IMG_WIDTH))
    return transformed_img

def transform_concat(train, labels, angle):
    list_aug_X = []
    list_aug_Y = []
    

    for i,img in tqdm(enumerate(train), total=train.shape[0]):
        list_aug_X.append(transform_img(img, angle))
        
    for i,img in tqdm(enumerate(labels), total=train.shape[0]):
        list_aug_Y.append(transform_img(img, angle))
    
    transformed_X = np.array(list_aug_X)
    transformed_Y = np.expand_dims(np.array(list_aug_Y), axis=3)
    
#     X = np.concatenate((train, transformed_X), axis=0)
#     Y = np.concatenate((labels, transformed_Y), axis=0)
    
    return transformed_X,transformed_Y

def flip_images(X_imgs):
    X_flip = []
    channels = X_imgs.shape[3]
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMG_HEIGHT, IMG_HEIGHT, channels))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i,img_data in tqdm(enumerate(X_imgs), total=X_imgs.shape[0]):
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img_data})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip

# shuffle two matrices in unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# all in one function
def data_aug(X_train, Y_train):
    print('Data augmentation')
    print('Rotate 90')
    X_tr, Y_tr = transform_concat(X_train, Y_train, 90)
    print('Rotate 180')
    X_tr_180, Y_tr_180 = transform_concat(X_train, Y_train, 180)
    print('Rotate 270')
    X_tr_270, Y_tr_270 = transform_concat(X_train, Y_train, 270)
    print('Flipping')
    X_flip = flip_images(X_train)
    Y_flip = flip_images(Y_train)
    print('Done data aug!')
    
    print('Concatenating the arrays:')
    rotated_X = np.concatenate((X_tr, X_tr_180, X_tr_270), axis=0)
    print('1')
    rotated_Y = np.concatenate((Y_tr, Y_tr_180, Y_tr_270), axis=0)
    print('1')
    X_auged = np.concatenate((rotated_X, X_flip), axis=0)
    print('1')
    Y_auged = np.concatenate((rotated_Y, Y_flip), axis=0)
    print('1')
    #X = np.concatenate((X_train, X_auged), axis=0)
    print('1')
    #Y = np.concatenate((Y_train, Y_auged), axis=0)
    print('Done with concatenating!')
    print('Shuffling the arrays')
    X_shuffled, Y_shuffled = unison_shuffled_copies(X_auged, Y_auged)
    return X_shuffled, Y_shuffled
