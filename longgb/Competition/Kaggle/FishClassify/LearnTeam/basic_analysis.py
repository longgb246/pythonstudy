#-*- coding:utf-8 -*-
import os
import numpy as np
import shutil


def splitData():
    '''
    Split the data into train and test.
    '''
    np.random.seed(2016)
    # save data
    root_train = r'D:\SelfLife\Competition\fish\train'
    root_val = r'D:\SelfLife\Competition\fish\test'
    # origin data
    root_total = r'D:\SelfLife\Competition\fish\org_train'
    # fish type
    FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    # count
    nbr_train_samples = 0
    nbr_val_samples = 0
    # Training proportion
    split_proportion = 0.8
    # split data
    for fish in FishNames:
        print fish
        if fish not in os.listdir(root_train):
            os.mkdir(os.path.join(root_train, fish))
        total_images = os.listdir(os.path.join(root_total, fish))
        nbr_train = int(len(total_images) * split_proportion)
        # shuffle the data
        np.random.shuffle(total_images)
        train_images = total_images[:nbr_train]
        val_images = total_images[nbr_train:]
        # get the train data
        for img in train_images:
            source = os.path.join(root_total, fish, img)
            target = os.path.join(root_train, fish, img)
            shutil.copy(source, target)
            nbr_train_samples += 1
        if fish not in os.listdir(root_val):
            os.mkdir(os.path.join(root_val, fish))
        # get the test data
        for img in val_images:
            source = os.path.join(root_total, fish, img)
            target = os.path.join(root_val, fish, img)
            shutil.copy(source, target)
            nbr_val_samples += 1
    # print
    print('Finish splitting train and val images!')
    print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))
    # training samples: 3019, # val samples: 758






