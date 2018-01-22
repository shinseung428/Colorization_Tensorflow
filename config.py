import tensorflow as tf
import os
import numpy as np
import scipy.misc
import argparse
import sys

#============================================================================================

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', True):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', False):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='')

#folder path setting
parser.add_argument('--data', dest='data', default='./data/', help='image datapath')
parser.add_argument('--images', dest='images', default='./images/', help='graph path')
parser.add_argument('--modelpath', dest='modelpath', default='./models/', help='model checkpoint path')
parser.add_argument('--graphpath', dest='graphpath', default='./graphs/', help='graph path')

#Image/Output setting
parser.add_argument('--input_width', dest='input_width', default=127, help='input image width')
parser.add_argument('--input_height', dest='input_height', default=63, help='input image height')


#Train setting
parser.add_argument('--continue_training', dest='continue_training', default=False, help='Flag to continue_training')

parser.add_argument('--epochs', dest='epochs', default=500, help='maximum training epoch')
parser.add_argument('--batch_size', dest='batch_size', default=64, help='batch size')

parser.add_argument('--learning_rate', dest='learning_rate', default=0.0001, help='learning_rate size')
parser.add_argument('--momentum', dest='momentum', default=0.5, help='momentum')

#Test Setting
parser.add_argument('--test_img', dest='test_img', default="", help='test image path')


args = parser.parse_args()

#============================================================================================
