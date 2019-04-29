import os

import time
import datetime
import numpy as np


import tensorflow as tf
from east import model
from east.eval import resize_image, sort_poly, detect


import collections

class Detector(object):

    def __init__(self, checkpoint_path):
        tf.reset_default_graph()
        self.input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        self.f_score, self.f_geometry = model.model(self.input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4,
                                                     inter_op_parallelism_threads=4))

        #ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
        model_path = checkpoint_path #os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(model_path))
        saver.restore(self.sess, model_path)

    def detect(self, img):
        start_time = time.time()
        rtparams = collections.OrderedDict()
        rtparams['start_time'] = datetime.datetime.now().isoformat()
        rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
        timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])

        im_resized, (ratio_h, ratio_w) = resize_image(img)
        rtparams['working_size'] = '{}x{}'.format(
            im_resized.shape[1], im_resized.shape[0])
        start = time.time()
        score, geometry = self.sess.run(
            [self.f_score, self.f_geometry],
            feed_dict={self.input_images: [im_resized[:, :, ::-1]]})
        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)

        if boxes is not None:
            scores = boxes[:, 8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        timer['overall'] = duration

        res_boxes = []
        if boxes is not None:
            for box, score in zip(boxes, scores):
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                flatten_box = box.flatten()
                res_boxes.append([min(int(flatten_box[0]), int(flatten_box[2])),
                                  min(int(flatten_box[1]), int(flatten_box[3])),
                                  max(int(flatten_box[4]), int(flatten_box[6])),
                                  max(int(flatten_box[5]), int(flatten_box[7]))])

        return res_boxes
