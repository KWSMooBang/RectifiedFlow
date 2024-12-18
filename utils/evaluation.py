import numpy as np
import jax
import six
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub


INCEPTION_TFHUB = "https://tfhub.dev/tensorflow/tfgan/eval/inception/1"
INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
_DEFAULT_DTYPES = {
    INCEPTION_OUTPUT: tf.float32,
    INCEPTION_FINAL_POOL: tf.float32
}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def get_inception_model(inceptionv3=False):
    if inceptionv3:
        return tfhub.load(
            "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"
        )
    else:
        return tfhub.load(INCEPTION_TFHUB)
    

def load_dataset_stats(config):
    """Load the pre-computed dataset statistics"""
    if config.data.dataset == 'CIFAR10':
        filename = "assets/stats/cifar10_stats.npz"
    elif config.data.dataset == 'CELEBA':
        filename = "assets/stats/celeba_stats.npz"
    elif config.data.dataset == 'LSUN':
        filename = f"assets/stats/lsun_{config.data.category}_{config.data.image_size}_stats.npz"
    else:
        raise ValueError(f"Dataset {config.data.dataset} stats not found.")

    with tf.io.gfile.GFile(filename, 'rb') as fin:
        stats = np.load(fin)
        return stats
    

def get_classifier_fn(output_fields, inception_model, return_tensor=False):
    """Return a function that can be as a classifier function

    Copied from tfgan but avoid loading the model each time calling classifier_fn

    Args:
        output_fields: A string, list, or 'None'. If present, assume the module
            outputs a dictionary, and select this field
        inception_model: A model loaded from TFHub
        return_tensor: If 'True', return a single tensor instead of a dictionary
    Returns:
        A one-argument function that takes an image Tensor and returns outputs
    """
    if isinstance(output_fields, six.string_types):
        output_fields = [output_fields]
        
    def classifier_fn(images):
        output = inception_model(images)
        if output_fields is not None:
            output = {x: output[x] for x in output_fields}
        if return_tensor:
            assert len(output) == 1
            output = list(output.values())[0]
        return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)
    
    return classifier_fn


@tf.function
def run_inception_jit(inputs, inception_model, num_batches=1, inceptionv3=False):
    """Running the inception network. Assuming input is within [0, 255]"""
    if not inceptionv3:
        inputs = (tf.cast(inputs, tf.float32) - 127.5) / 127.5
    else:
        inputs = tf.cast(inputs, tf.float32) / 255.
        
    return tfgan.eval.run_classifier_fn(
        inputs,
        num_batches=num_batches,
        classifier_fn=get_classifier_fn(None, inception_model),
        dtypes=_DEFAULT_DTYPES
    )
    

@tf.function
def run_inception_distributed(input_tensor, inception_model, num_batches=1, inceptionv3=False):
    """Distribute the inception network computation to all available TPUs
    
    Args:
        input_tensor: The input images. Assumed to be within [0, 255]
        inception_model: The inception network model obtained from 'tfhub'
        num_batches: The number of batches used for dividing the input
        inceptionv3: If 'True', use InceptionV3, otherwise use InceptionV1
    Returns:
        A dictionary with key 'pool_3' and 'logits', representing the pool_3 and
            logits of the inception network respectively
    """
    num_tpus = jax.local_device_count()
    input_tensors = tf.split(input_tensor, num_tpus, axis=0)
    pool3 = []
    logits = [] if not inceptionv3 else None
    device_format = '/TPU:{}' if 'TPU' in str(jax.devices()[0]) else '/GPU:{}'
    for i, tensor in enumerate(input_tensors):
        with tf.device(device_format.format(i)):
            tensor_on_device = tf.identity(tensor)
            results = run_inception_jit(
                tensor_on_device, inception_model, num_batches=num_batches,
                inceptionv3=inceptionv3
            )
            
            if not inceptionv3:
                pool3.append(results['pool_3'])
                logits.append(results['logits'])
            else:
                pool3.append(results['pool_3'])
            
        with tf.device('/CPU'):
            return {
                'pool_3': tf.concat(pool3, axis=0),
                'logits': tf.concat(logits, aixs=0) if not inceptionv3 else None
            }
    
    