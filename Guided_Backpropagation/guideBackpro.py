# File guideBackpro.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

from tensorcv.models.layers import global_avg_pool

@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    gate_g = tf.cast(grad > 0, "float32")
    gate_y = tf.cast(op.outputs[0] > 0, "float32")
    return grad * gate_g * gate_y

class GuideBackPro(object):
    def __init__(self, vis_model=None, class_id=None):
        assert not vis_model is None, 'vis_model cannot be None!'
        assert not class_id is None, 'class_id cannot be None!'

        self._vis_model = vis_model
        self._class_id = class_id

        self._GuidedReluRegistered = False

    def _create_model(self, image):
        keep_prob = 1
        self._vis_model.create_model([image, keep_prob])

        out_act = global_avg_pool(self._vis_model.layer['output'])
        nclass = out_act.shape.as_list()[-1]

        one_hot = tf.sparse_to_dense([[1, 0]], [nclass, 1], 1.0)
        out_act = tf.reshape(out_act, [1, nclass])
        self.class_act = tf.matmul(out_act, one_hot)
        self.pre_label = tf.nn.top_k(tf.nn.softmax(out_act), k=5, sorted=True)

    def create_graph(self, image):
        g = tf.get_default_graph()
        with g.gradient_override_map({'Relu': 'GuidedRelu'}):
            self._create_model(image)
            self.guided_grads_node = tf.gradients(self.class_act, 
                                                     self._vis_model.layer['input'])
