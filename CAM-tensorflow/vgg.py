# File: vgg.py
# Author: Qian Ge <geqian1001@gmail.com>
import argparse

import tensorflow as tf

import tensorcv
from tensorcv.dataflow.image import *
from tensorcv.train.config import TrainConfig
from tensorcv.train.simple import SimpleFeedTrainer
from tensorcv.callbacks import *

from model import VGGCAM
import configvgg as config_path

NUM_CLASS = 257

def get_config(FLAGS):
    dataset_train = ImageLabelFromFolder('.jpg', data_dir = config_path.data_dir, 
                        num_class = NUM_CLASS,
                        reshape = 224,
                        num_channel = 3)

    # dataset_val = MNISTLabel('val', config_path.data_dir)
    # dataset_test = ImageFromFile('.png', 
    #                             data_dir = config_path.data_dir, 
    #                             shuffle = False,
    #                             normalize_fnc = normalize_one,
    #                             num_channel = 1)

    inference_list = [InferScalars('accuracy/result', 'test_accuracy')]
    infer_list = InferImages('classmap/result','image', color = True)
    return TrainConfig(
                 dataflow = dataset_train, 
                 model = VGGCAM(num_class = NUM_CLASS, 
                           inspect_class = None,
                           learning_rate = 0.0001,
                           is_load = True,
                           pre_train_path = 'E:\\GITHUB\\workspace\\CNN\pretrained\\vgg19.npy'),
                 monitors = TFSummaryWriter(),
                 callbacks = [
                    ModelSaver(periodic = 100),
                    TrainSummary(key = 'train', periodic = 10),
                    # FeedInferenceBatch(dataset_val, periodic = 100, batch_count = 100, 
                    #               # extra_cbs = TrainSummary(key = 'test'),
                    #               inferencers = inference_list),
                    # FeedInference(dataset_test, periodic = 100,
                    #               infer_batch_size = 1, 
                    #               inferencers = infer_list),
                    CheckScalar(['accuracy/result','loss/result'], periodic = 10),
                  ],
                 batch_size = FLAGS.batch_size, 
                 max_epoch = 100,
                 summary_periodic = 100,
                 default_dirs = config_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 1, type = int)
    parser.add_argument('--label', default = 1, type = int,
                        help = 'Label of inspect class.')

    parser.add_argument('--predict', action = 'store_true', 
                        help = 'Run prediction')
    parser.add_argument('--train', action = 'store_true', 
                        help = 'Train the model')

    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = get_args()
    config = get_config(FLAGS)
    SimpleFeedTrainer(config).train()

    # num_class = 257
    # num_channels = 3
    # batch_size = 1


    # vgg_cam_model = VGGCAM(num_class = num_class, 
    #                        inspect_class = None,
    #                        num_channels = num_channels, 
    #                        learning_rate = 0.0001,
    #                        is_load = True,
    #                        pre_train_path = 'E:\\GITHUB\\workspace\\CNN\pretrained\\vgg19.npy')
    

    # data_train = ImageLabelFromFolder('.jpg', data_dir = '', 
    #                     num_channel = num_channels,
    #                     label_dict = None, num_class = num_class,
    #                     one_hot = False,
    #                     shuffle = True,
    #                     reshape = 224)
    # data_train.set_batch_size(batch_size)
    
    # # print((data_train.next_batch()).shape)
    
    # vgg_cam_model.create_graph()
    # im_plh, label_plh = vgg_cam_model.get_train_placeholder()[0],\
    #                       vgg_cam_model.get_train_placeholder()[1] 

    # # tf.image.resize_images(tf.cast(image, tf.float32), 
    #     # [tf.cast(new_height, tf.int32), tf.cast(new_width, tf.int32)])

    # grads = vgg_cam_model.get_grads()
    # opt = vgg_cam_model.get_optimizer()
    # train_op = opt.apply_gradients(grads, name = 'train')

    # summary_list = tf.summary.merge_all('default')

    # writer = tf.summary.FileWriter('E:\\GITHUB\\workspace\\CNN\\other\\')
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     writer.add_graph(sess.graph)

    #     batch_data = data_train.next_batch()
    #     sess.run(train_op, feed_dict = {im_plh: batch_data[0], label_plh: batch_data[1], vgg_cam_model.keep_prob:0.5})
    # writer.close()

    # print(tf.trainable_variables())
    # # optimizer = vgg_cam_model.get_optimizer()
    # # loss = vgg_cam_model.get_loss()
    # # grads = optimizer.compute_gradients(loss)
    # # grads = vgg_cam_model.get_grads()
    # # opt = vgg_cam_model.get_optimizer()
    # # train_op = opt.apply_gradients(grads, name = 'train')



    # writer = tf.summary.FileWriter('D:\\Qian\\GitHub\\workspace\\test\\')
    # with tf.Session() as sess:
    #     # sess.run(tf.global_variables_initializer())
    #     writer.add_graph(sess.graph)

    # writer.close()


# def overlay(img, heatmap, cmap='jet', alpha=0.5):

#     if isinstance(img, np.ndarray):
#         img = Image.fromarray(img)

#     if isinstance(heatmap, np.ndarray):
#         colorize = plt.get_cmap(cmap)
#         # Normalize
#         heatmap = heatmap - np.min(heatmap)
#         heatmap = heatmap / np.max(heatmap)
#         heatmap = colorize(heatmap, bytes=True)
#         heatmap = Image.fromarray(heatmap[:, :, :3], mode='RGB')

#     # Resize the heatmap to cover whole img
#     heatmap = heatmap.resize((img.size[0], img.size[1]), resample=Image.BILINEAR)
#     # Display final overlayed output
#     result = Image.blend(img, heatmap, alpha)
#     return result

