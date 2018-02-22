from __future__ import print_function

import click
import os
import time
import cv2
import scipy.io as sio
import tensorflow as tf
import numpy as np
from PIL import Image
from model import DeepLabResNetModel

import sys
sys.path.append("/home/brainoft/learning/offline_learning")
from visualizations.image_viewer import ImageIndex
from learning.schedules import get_host_configs

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

NUM_CLASSES = 27
SAVE_DIR = './output/'
MODEL_DIR = './restore_weights/'
matfn = 'color150.mat'


@click.group()
def cli():
    pass


def read_labelcolours(matfn):
    mat = sio.loadmat(matfn)
    color_table = mat['colors']
    shape = color_table.shape
    color_list = [tuple(color_table[i]) for i in range(shape[0])]

    return color_list


def decode_labels(mask, num_images=1, num_classes=150):
    label_colours = read_labelcolours(matfn)

    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


@click.command()
@click.argument('hostname', type=str)
@click.argument('img_dir', type=click.Path(file_okay=False))
@click.option('--output_dir', '-o', default=None, type=click.Path(file_okay=False),
              help="Directory where output should be stored")
@click.option('--begin_time', '-b', type=float, default=None, help="start time in epoch seconds")
@click.option('--end_time', '-e', type=float, default=None, help="stop time in epoch seconds")
@click.option('--step_size', '-s', type=float, default=0.5, help="step size in seconds")
@click.option("--camera_list", "-c", type=str, default=None,
              help="Comma separated list of cameras for which heatmap should be generated")
def semantic_segment(hostname, img_dir, output_dir, begin_time, end_time, step_size, camera_list):
    host_configs = get_host_configs(hostname)
    cameras = camera_list.split(",")
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create image index
    print("Creating image index")
    start = time.time()
    image_index = ImageIndex(img_dir)
    print("Image index created in {} secs".format(time.time() - start))

    for camera in cameras:
        camera_output_dir = os.path.join(output_dir, camera) if len(cameras) > 1 else output_dir
        if not os.path.exists(camera_output_dir):
            os.makedirs(camera_output_dir)
        for t_sec in np.arange(begin_time, end_time + step_size, step_size):
            print("\nProcessing {} {}".format(camera, t_sec))
            start = time.time()
            img_path = image_index.image_filename(camera, t_sec)
            file_type = img_path.split('.')[-1]
            # Prepare image.
            if file_type.lower() == 'png':
                img = tf.image.decode_png(tf.read_file(img_path), channels=3)
            elif file_type.lower() == 'jpg':
                img = tf.image.decode_jpeg(tf.read_file(img_path), channels=3)
            else:
                print('cannot process {0} file.'.format(file_type))
                continue

            # Convert RGB to BGR.
            img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
            img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
            # Extract mean.
            img -= IMG_MEAN

            # TODO(kunal): Figure out how to update network input so that we don't have to create new one in loop
            # Set up TF session and initialize variables.
            network_start = time.time()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                data = {'data': tf.expand_dims(img, dim=0)}
                net = DeepLabResNetModel(data, is_training=False, num_classes=NUM_CLASSES)
                # Which variables to load.
                restore_var = tf.global_variables()
                # Predictions.
                raw_output = net.layers['fc_out']
                raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2, ])
                raw_output_up = tf.argmax(raw_output_up, dimension=3)
                pred = tf.expand_dims(raw_output_up, dim=3)
                # Load weights.
                ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
                if ckpt and ckpt.model_checkpoint_path:
                    loader = tf.train.Saver(var_list=restore_var)
                    load(loader, sess, ckpt.model_checkpoint_path)
                else:
                    print('No checkpoint file found.')
                    load_step = 0
                print("Network created in {} secs".format(time.time() - network_start))

                # Perform inference.
                inference_start = time.time()
                preds = sess.run(pred)
                print("Inference done in {} secs".format(time.time() - inference_start))

                msk = decode_labels(preds, num_classes=NUM_CLASSES)

                im = Image.fromarray(msk[0])
                opencv_im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                if output_dir:
                    cv2.imwrite(os.path.join(camera_output_dir, '{}.jpg'.format(int(np.around(t_sec * 1000)))), opencv_im)
                else:
                    cv2.imshow("Output", opencv_im)
                    cv2.imshow("Input", cv2.imread(img_path))
                    cv2.waitKey(50)
            tf.reset_default_graph()
            print("Processed {} in {} secs".format(t_sec, time.time() - start))


if __name__ == '__main__':
    cli.add_command(semantic_segment)
    cli()
