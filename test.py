import tensorflow as tf
import os
import numpy as np
import scipy.misc
import argparse
import sys

from config import *
from network import network
import cv2

def test(args, sess, model):
    #optimizer
    optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer").minimize(model.loss, var_list=model.trainable_vars)

    epoch = 0
    step = 0
    overall_step = 0

    #saver
    saver = tf.train.Saver()        
    
    last_ckpt = tf.train.latest_checkpoint(args.modelpath)
    saver.restore(sess, last_ckpt)
    ckpt_name = str(last_ckpt)
    print "Loaded model file from " + ckpt_name
    

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    
    writer = tf.summary.FileWriter(args.graphpath, sess.graph)

    res_rgb = sess.run(model.pred_rgb)
    cv2.imshow("result", res_rgb)
    cv2.waitKey(1)

    coord.request_stop()
    coord.join(threads)
    sess.close()            
    print("Done.")


def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    
    with tf.Session(config=run_config) as sess:
        model = network(args)

        #create graph and checkpoints folder if they don't exist
        if not os.path.exists(args.modelpath):
            os.makedirs(args.modelpath)
        if not os.path.exists(args.graphpath):
            os.makedirs(args.graphpath)
            

        
        print 'Start Training...'
        test(args, sess, model)


main(args)

#Still Working....