import tensorflow as tf
import os
import numpy as np
import scipy.misc
import argparse
import sys

from config import *
from DataReader import *
from network import network
import cv2

def test(args, sess, model):
    image, image_3 = load_test_data(args)

    #saver
    saver = tf.train.Saver()        
    
    last_ckpt = tf.train.latest_checkpoint(args.modelpath)
    saver.restore(sess, last_ckpt)
    ckpt_name = str(last_ckpt)
    print "Loaded model file from " + ckpt_name
    

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    
    writer = tf.summary.FileWriter(args.graphpath, sess.graph)

    # res_rgb = sess.run(model.pred_rgb)
    res_rgb = sess.run([model.pred_rgb], feed_dict={model.gray: image, model.orig: image_3})
    res_image = res_rgb[0][0]

    cv2.imshow("result", res_image)
    cv2.waitKey()

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
            

        
        print 'Processing Image...'
        test(args, sess, model)


main(args)

#Still Working....