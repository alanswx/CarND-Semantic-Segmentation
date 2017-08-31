import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
   
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    return tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name), tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name), tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name), tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name), tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

 
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    #tf.layers.conv2d(x, num_outputs, 1, 1, weights_initializer=custom_init)

    # deconvolute from layer 7
    layer7_onexone=tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1))
    layer7_upsample=tf.layers.conv2d_transpose(layer7_onexone,num_classes,4,strides=(2,2),padding='same')

    # create skip layers from 4 to 7 (add it)
    layer4_onexone=tf.layers.conv2d(vgg_layer4_out,num_classes,1,strides=(1,1)) 
    layer4=tf.add(layer7_upsample,layer4_onexone)
    # transpose it up another 2x
    layer4_upsample=tf.layers.conv2d_transpose(layer4,num_classes,4,strides=(2,2),padding='same')

    # create skip layers from 3 to current (add it)
    layer3_onexone=tf.layers.conv2d(vgg_layer3_out,num_classes,1,strides=(1,1)) 
    layer3=tf.add(layer4_upsample,layer3_onexone)
    # transpose it up again
    output = tf.layers.conv2d_transpose(layer3,num_classes,16,strides=(8,8),padding='same',name='output')
    

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_labels_r = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_labels_r, logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    # setup globals
    sess.run(tf.global_variables_initializer())

    for epoch in range (epochs):
         batch_index = 0
         for batch_x,batch_y in get_batches_fn(batch_size):
            batch_index=batch_index+1
            _,loss=sess.run( [train_op, cross_entropy_loss], feed_dict= {input_image: batch_x, correct_label: batch_y, keep_prob: 0.5 , learning_rate: 0.001 } )
         print(" Epoch: {} | loss: {}".format(epoch,loss))
   
    pass
tests.test_train_nn(train_nn)


def run():
    print("run")
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    num_epochs=12 #0.09
    #num_epochs=24
    batch_size=8
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        correct_label = tf.placeholder(tf.float32,(None,image_shape[0],image_shape[1],num_classes))
        learning_rate = tf.placeholder(tf.float32,())
        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        #sess.run(image_input)
        output_layer = layers(layer3_out,layer4_out,layer7_out,num_classes)
        logits,train_op,cross_entropy_loss=optimize(output_layer,correct_label,learning_rate,num_classes)

        writer = tf.summary.FileWriter("output", sess.graph)
        # TODO: Train NN using the train_nn function
        train_nn(sess,num_epochs,batch_size,get_batches_fn,train_op,cross_entropy_loss,input_image,correct_label,keep_prob,learning_rate)
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        writer.close()
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
