from scipy import misc
import scipy.io

import tensorflow as tf

from tensorcv.dataflow.common import *

from common.models import VGG
from Guided_Backpropagation.guideBackpro import GuideBackPro

IM_PATH = 'E:\\GITHUB\\workspace\\CNN\\dataset\\test\\ILSVRC2017_test_00000004.jpg'
save_dir = 'E:\\GITHUB\\workspace\\CNN\\test\\'
vgg_path = 'E:\\GITHUB\\workspace\\CNN\\pretrained\\vgg19.npy'

if __name__ == '__main__':

    image = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    im = load_image(IM_PATH, read_channel=3)
    model = GuideBackPro(vis_model=VGG.VGG19_FCN(is_load=True, pre_train_path=vgg_path, is_rescale=True), class_id=1)
    model.create_graph(image)    

    writer = tf.summary.FileWriter(save_dir)
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        guided_backpro, label = sess.run([model.guided_grads_node, model.pre_label], feed_dict = {image: im})
        print(label)
        scipy.misc.imsave(save_dir + 'test1.png', np.squeeze(guided_backpro))

    writer.close()
    


