from moviepy.editor import VideoFileClip
import tensorflow as tf
import numpy as np
import os.path
import scipy.misc


import main

tf.reset_default_graph()
image_shape = (160, 576)

# Add ops to save and restore all the variables.

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
sess = tf.Session()
num_classes = 2
data_dir = './data'
runs_dir = './runs'

vgg_path = os.path.join(data_dir, 'vgg')
# Create function to get batches
correct_label = tf.placeholder(tf.float32,(None,image_shape[0],image_shape[1],num_classes))
learning_rate = tf.placeholder(tf.float32,())
# TODO: Build NN using load_vgg, layers, and optimize function
input_image, keep_prob, layer3_out, layer4_out, layer7_out = main.load_vgg(sess, vgg_path)
#sess.run(image_input)
output_layer = main.layers(layer3_out,layer4_out,layer7_out,num_classes)
logits,train_op,cross_entropy_loss=main.optimize(output_layer,correct_label,learning_rate,num_classes)

saver = tf.train.Saver()
# Restore variables from disk.
saver.restore(sess, "model.ckpt")
print("Model restored.")

image_file = "data/data_road/testing/image_2/uu_000099.png"
image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_image: [image]})
im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
mask = scipy.misc.toimage(mask, mode="RGBA")
street_im = scipy.misc.toimage(image)
street_im.paste(mask, box=None, mask=mask)

scipy.misc.imsave("out.png",street_im)

def predict(image):
  image = scipy.misc.imresize(image,image_shape)
  image = np.array(image)  
  im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_image: [image]})
  im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
  segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
  mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
  mask = scipy.misc.toimage(mask, mode="RGBA")
  street_im = scipy.misc.toimage(image)
  street_im.paste(mask, box=None, mask=mask)

  street_im= np.array(street_im)  
  return street_im



def processVideo(input_video,output):
  clip1 = VideoFileClip(input_video)
  print("about to predict on video",input_video)
  out_clip = clip1.fl_image(predict)
  out_clip.write_videofile(output,audio=False)


if __name__ == '__main__':
    #processVideo('project_video.mp4','project_video_out.mp4')
    #processVideo('challenge_video.mp4','challenge_video_out.mp4')
    processVideo('harder_challenge_video.mp4','harder_challenge_video_out.mp4')

