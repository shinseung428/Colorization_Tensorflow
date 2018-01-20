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
parser.add_argument('--datapath', dest='datapath', default='./data/', help='image datapath')
parser.add_argument('--modelpath', dest='modelpath', default='./modelpath/', help='model checkpoint path')
parser.add_argument('--graphpath', dest='graphpath', default='./graphpath/', help='graph path')

#Image/Output setting
parser.add_argument('--input_width', dest='input_width', default=224, help='input image width')
parser.add_argument('--input_height', dest='input_height', default=224, help='input image height')


#Train setting
parser.add_argument('--continue_training', dest='continue_training', default=False, help='Flag to continue_training')

parser.add_argument('--epochs', dest='epoch', default=10, help='maximum training epoch')
parser.add_argument('--batch_size', dest='batch_size', default=32, help='batch size')

parser.add_argument('--learning_rate', dest='learning_rate', default=0.0001, help='learning_rate size')
parser.add_argument('--momentum', dest='momentum', default=0.5, help='momentum')

#Test Setting


args = parser.parse_args()

#============================================================================================
