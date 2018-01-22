import tensorflow as tf
import os
import numpy as np
import scipy.misc
import argparse
import sys

from config import *
from network import network
from DataReader import *
from utils import *

def train(args, sess, model):
    v_images = load_valid_data(args)


    #optimizer
    optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer").minimize(model.loss, var_list=model.trainable_vars)

    epoch = 0
    step = 0
    overall_step = 0

    #saver
    saver = tf.train.Saver()        
    if args.continue_training:
        last_ckpt = tf.train.latest_checkpoint(args.modelpath)
        saver.restore(sess, last_ckpt)
        ckpt_name = str(last_ckpt)
        print "Loaded model file from " + ckpt_name
        step = int(ckpt_name.split('-')[-1])
    else:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    all_summary = tf.summary.merge([model.loss_sum,
                                    model.input_img_sum,
                                    model.gray_img_sum,
                                    model.gt_lab_sum,
                                    model.pred_lab_sum,
                                    model.pred_rgb_sum
                                   ])

    
    writer = tf.summary.FileWriter(args.graphpath, sess.graph)

    while epoch < args.epochs:
        summary, loss, _ = sess.run([all_summary, 
                                     model.loss, 
                                     optimizer])
        writer.add_summary(summary, overall_step)

        print "step [%d] Training Loss: [%.4f] " % (step, loss)

        step += 1
        overall_step += 1

        if step % 1000 == 0:
            saver.save(sess, args.modelpath + "model", global_step=step)
            print "Model saved at step %s" % str(step)                
            step += 1

            res = sess.run([model.pred_rgb], feed_dict={model.gray:v_images})
            img_tile(step, args, res)


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
        train(args, sess, model)


main(args)

#Still Working....