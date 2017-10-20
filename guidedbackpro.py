from scipy import misc
import scipy.io

import tensorflow as tf
import numpy as np
from tensorcv.dataflow.image import *

from common.models import VGG
from Guided_Backpropagation.guideBackpro import GuideBackPro

IM_PATH = 'E:\\GITHUB\\workspace\\CNN\\dataset\\test2\\'
SAVE_DIR = 'E:\\GITHUB\\workspace\\CNN\\test\\'
VGG_PATH = 'E:\\GITHUB\\workspace\\CNN\\pretrained\\vgg19.npy'

if __name__ == '__main__':

    image = tf.placeholder(tf.float32, shape=[None, None, None, 3])

    input_im = ImageFromFile('.jpg', data_dir=IM_PATH, 
                             num_channel=3, shuffle=False)
    input_im.set_batch_size(1)

    # im = load_image(IM_PATH, read_channel=3)
    model = GuideBackPro(vis_model=VGG.VGG19_FCN(is_load=True, 
                                                 pre_train_path=VGG_PATH, 
                                                 is_rescale=True))
    back_pro_op = model.comp_guided_backpro(image)    

    writer = tf.summary.FileWriter(SAVE_DIR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        cnt = 0
        while input_im.epochs_completed < 1:
            im = input_im.next_batch()[0]
            guided_backpro, label, o_im = sess.run([back_pro_op, model.pre_label, model.input_im], 
                                             feed_dict = {image: im})
            print(label)
            for cid, guided_map in zip(guided_backpro[1], guided_backpro[0]):
                scipy.misc.imsave('{}map_{}_class_{}.png'.format(SAVE_DIR, cnt, cid), 
                                  np.squeeze(guided_map))
            scipy.misc.imsave('{}im_{}.png'.format(SAVE_DIR, cnt), 
                              np.squeeze(o_im))
                # scipy.io.savemat('{}map_1_class_{}.mat'.format(SAVE_DIR, cid), 
                #                  {'mat': np.squeeze(guided_map)*255})
            cnt += 1

    writer.close()
    


