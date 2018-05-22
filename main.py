import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_bool('TRANSFER',True, "Bottleneck features training file (.p)")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 5, "The batch size.")
flags.DEFINE_float('learning_rate',0.0001,"Learning Rate")
flags.DEFINE_float('kp',0.75,"Keep Probability")
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
    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
# tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    print("inside layer function")
    if FLAGS.TRANSFER:
        tf.stop_gradient(vgg_layer7_out)
        tf.stop_gradient(vgg_layer4_out)
        tf.stop_gradient(vgg_layer3_out)

    vgg_layer4_out = tf.Print(vgg_layer4_out, [tf.shape(vgg_layer4_out)], message="Shape of layer 4 at start",
                            summarize=4, first_n=1)

    vgg_layer7_out = tf.Print(vgg_layer7_out, [tf.shape(vgg_layer7_out)], message="Shape of layer 7 at start",
                            summarize=4, first_n=1)

    vgg_layer3_out = tf.Print(vgg_layer3_out, [tf.shape(vgg_layer3_out)], message="Shape of layer 3 at start",
                            summarize=4, first_n=1) 

    with tf.variable_scope("decoder"):
        layer7_1x1=tf.layers.conv2d(vgg_layer7_out,num_classes,1,strides=(1,1),
                                    kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                    name="new_7_1x1")
        
        layer4_1x1=tf.layers.conv2d(vgg_layer4_out,num_classes,1,strides=(1,1),
                                    kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                    name="new_4_1x1")
        layer3_1x1=tf.layers.conv2d(vgg_layer3_out,num_classes,1,strides=(1,1),
                                    kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                    name="new_3_1x1")
        layer7_dconv=tf.layers.conv2d_transpose(layer7_1x1,num_classes,4,(2,2), padding="same",
                                                kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                                kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                                name="new_7_deconv")
        layer7_dconv = tf.Print(layer7_dconv, [tf.shape(layer7_dconv)], message="Shape of layer 7 after 2x upsampling",
                                first_n=1, name="new_7_deconv_print")                                        
        layer4_7add=tf.add(layer7_dconv,layer4_1x1)
        layer4_7_dconv=tf.layers.conv2d_transpose(layer4_7add,num_classes,4,(2,2), padding="same",
                                                    kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                                    name="new_47_deconv")
        layer4_7_3add=tf.add(layer4_7_dconv,layer3_1x1)
        layer473_deconv=tf.layers.conv2d_transpose(layer4_7_3add,num_classes,16,(8,8),padding="same",
                                                    kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                                    name="new_473_deconv")

        layer473_deconv = tf.Print(layer473_deconv, [tf.shape(layer473_deconv)], message="Shape of final layer",
                            summarize=4, first_n=1) 


    #print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"decoder"))       
    
    return layer473_deconv
# tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = nn_last_layer,labels = correct_label)
    mean_cross_entropy = tf.reduce_mean(cross_entropy)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope="decoder")
    reg_constant = 0.1  # Choose an appropriate one.
    mean_cross_entropy = mean_cross_entropy + reg_constant * sum(reg_losses)
    opt = None
    with tf.variable_scope("decoder"):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # training_op=None
    if FLAGS.TRANSFER:
        trainable_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="decoder")
        #print(len(trainable_variables))
        trainable_variables += [var for var in tf.global_variables() if 'beta' in var.name]
        print("Trainable variables:", len(trainable_variables))
        training_op = opt.minimize(mean_cross_entropy,var_list=trainable_variables,name="training_op")
        return nn_last_layer,training_op,mean_cross_entropy
    else:
        training_op = opt.minimize(mean_cross_entropy,name="training_op")
        return nn_last_layer,training_op,mean_cross_entropy
        
    
#tests.test_optimize(optimize)


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
    for epoch in range(epochs):
        print(epoch)
        for images,labels in get_batches_fn(batch_size):
            loss,_ = sess.run([cross_entropy_loss,train_op],feed_dict={input_image:images,correct_label:labels,keep_prob:FLAGS.kp,learning_rate:FLAGS.learning_rate})
            # if loss_flag:
            #     temp_loss = loss
            #     loss_flag = 1
                
            print("Loss:- ",loss)
# tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    # tests.test_for_kitti_dataset(data_dir)

    tf.reset_default_graph()

    # Download pretrained vgg model
    # helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        print("generating batch function")
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        
        image_input, keep_prob, layer3_out, layer4_out, layer7_out=load_vgg(sess,vgg_path)
        epochs = FLAGS.epochs
        # learning_rate_f = FLAGS.learning_rate
        batch_size = FLAGS.batch_size

        final_layer = layers(layer3_out,layer4_out,layer7_out,num_classes)
        #print("trainable variables after layer function")
        #print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder"))
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')

        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        print("in optimizing")
        logits, train_op, cross_entropy_loss = optimize(final_layer, correct_label, learning_rate, num_classes)
        print("in training")
        if FLAGS.TRANSFER and True:
            trainable_variables = tf.global_variables(scope="decoder")
            trainable_variables += [var for var in tf.global_variables() if 'beta' in var.name]
            print(trainable_variables)
            my_variable_init = [var.initializer for var in trainable_variables]
            sess.run(my_variable_init)
        else:
            sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,correct_label, keep_prob, learning_rate)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)



if __name__ == '__main__':
    run()
