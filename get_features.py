import scipy.io as sio
import vgg16
import tensorflow as tf
import utils
import cv2
import numpy as np
import skimage
from skimage import io
from skimage import transform
import pandas as pd
from scipy.linalg import norm
import os

batch_size = 32
annotation_path = 'D:/dataset_code/数据集/flickr+mscoco/flickr30k/results_20130124.token'
flickr_image_path = 'D:/dataset_code/数据集/flickr+mscoco/flickr30k/flickr30k-images/'
feat_path = 'D:/dataset_code/数据集/flickr+mscoco/flickr30k/data/vgg16_feats.npy'
annotation_result_path = 'D:/dataset_code/数据集/flickr+mscoco/flickr30k/data/annotations.pickle'
annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path, x.split('#')[0]))

#获取文件夹下每一张图片
unique_images = annotations['image'].unique()
#print(len(unique_images))#31783
image_df = pd.DataFrame({'image': unique_images, 'image_id': range(len(unique_images))})
# 每张图片对应5个句子
annotations = pd.merge(annotations, image_df)
annotations.to_pickle(annotation_result_path)

def get_feats():
    vgg16_feats = np.zeros((len(unique_images), 4096))
    with tf.Session() as sess:
        images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
        vgg = vgg16.Vgg16()
        vgg.build(images)
        for i in range(len(unique_images)):
            img_list = utils.load_image(unique_images[i])
            batch = img_list.reshape((1, 224, 224, 3))
            feature = sess.run(vgg.fc7, feed_dict={images: batch})#提取fc7层的特征
            feature = np.reshape(feature, [4096])
	    feature /= norm(feature) # 特征归一化
            vgg16_feats[i, :] = feature #每张图片的特征向量为1行
    vgg16_feats = np.save('D:/dataset_code/数据集/flickr+mscoco/flickr30k/data/vgg16_feats', vgg16_feats)
    return vgg16_feats


if __name__ == '__main__':
    get_feats()
