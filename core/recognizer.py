import cv2
import numpy as np
import tensorflow as tf

from CRNN import config
from CRNN.models import crnn_net
from CRNN import tf_io_pipline_fast_tools

CFG = config.cfg


class Recognizer(object):

    def __init__(self, model):
        tf.reset_default_graph()

        self.width = 100
        self.height = 32

        self.inputdata = tf.placeholder(
            dtype=tf.float32,
            shape=[1, self.height, self.width, CFG.ARCH.INPUT_CHANNELS],
            name='input'
        )

        self.codec = tf_io_pipline_fast_tools.CrnnFeatureReader(
            char_dict_path="char_dict_en.json",
            ord_map_dict_path="ord_map_en.json"
        )

        net = crnn_net.ShadowNet(
            phase='test',
            hidden_nums=CFG.ARCH.HIDDEN_UNITS,
            layers_nums=CFG.ARCH.HIDDEN_LAYERS,
            num_classes=CFG.ARCH.NUM_CLASSES
        )

        inference_ret = net.inference(
            inputdata=self.inputdata,
            name='shadow_net',
            reuse=False
        )

        self.decodes, _ = tf.nn.ctc_beam_search_decoder(
            inputs=inference_ret,
            sequence_length=int(self.width / 4) * np.ones(1),
            merge_repeated=False,
            beam_width=10
        )

        # config tf saver
        saver = tf.train.Saver()

        # config tf session
        sess_config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4,
                                     inter_op_parallelism_threads=4)
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH

        self.sess = tf.Session(config=sess_config)
        saver.restore(sess=self.sess, save_path=model)

    def recognize(self, img):
        #image = cv2.imread(img, cv2.IMREAD_COLOR)
        image = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        image = np.array(image, np.float32) / 127.5 - 1.0

        preds = self.sess.run(self.decodes, feed_dict={self.inputdata: [image]})

        return self.codec.sparse_tensor_to_str(preds[0])[0]


