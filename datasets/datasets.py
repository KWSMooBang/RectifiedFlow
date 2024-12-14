import os
import jax
import tensorflow as tf
import tensorflow_datasets as tfds


def get_data_scaler(config):
    """
    Data normalizer
    Assume data are always in [0, 1]
    """
    if config.data.centered:
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x
    

def get_data_inverse_scaler(config):
    """
    Inverse data normalizer
    """
    if config.data.centered:
        return lambda x: (x + 1.) * 2.
    else:
        return lambda x: x
    

def crop_resize(image, resolution):
    """
    Crop and resize an image to the given resolution
    """
    crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    image = image[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
    image = tf.image.resize(
        image,
        size=(resolution, resolution),
        antialias=True,
        method=tf.image.ResizeMethod.BICUBIC
    )
    return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
    """
    Shrink an image to the given resolution
    """
    h, w = image.shape[0], image.shape[1]
    ratio = resolution / min(h, w)
    h = tf.round(h * ratio, tf.int32)
    w = tf.round(w * ratio, tf.int32)
    return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
    """
    crop the center of an image to the given size
    """
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_data(config, uniform_dequantization=False, evaluation=False):
    """Create data loaders for training and evaluation

    Args:
        config: A ml_collection.ConfigDict parsed from config files
        uniform_dequantization: If 'True', add uniform dequantization to images
        evaluation: If 'True', fix number of epochs to 1
    Returns:
        train_ds, eval_ds, dataset_handler
    """
    if 'pytorch' in config.data.dataset.lower():
        train_ds, eval_ds = get_pytorch_dataset(config)
        return train_ds, eval_ds

    import resource
    low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
    
    # Compute batch size for this worker
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    if batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch sizes ({batch_size}) must be divided by the number of devices ({jax.device_count()})")

    # Reduce this when image resolution is too large and data pointer is stored
    shuffle_buffer_size = 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1
    
    # Create dataset builders for each dataset
    if config.data.dataset == 'CIFAR10':
        dataset_builder = tfds.builder('cifar10')
        train_split_name = 'train'
        eval_split_name = 'eval'
        
        def resize_op(image):
            image = tf.image.convert_image_dtype(image, tf.float32)
            return tf.image.resize(image, [config.data.image_size, config.data.image_size], antialias=True)
    
    elif config.data.dataset == 'SVHN':
        dataset_builder = tfds.builder('svhn_cropped')
        train_split_name = 'train'
        eval_split_name = 'test'
        
        def resize_op(image):
            image = tf.image.convert_image_dtype(image, tf.float32)
            return tf.image.resize(image, [config.data.image_size, config.data.image_size], antialias=True)
    
    elif config.data.dataset == 'CELEBA':
        dataset_builder = tfds.builder('celeb_a', data_dir=config.data.root_path)
        train_split_name = 'train'
        eval_split_name = 'validation'
        
        def resize_op(image):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = central_crop(image, 140)
            image = resize_small(image, config.data.image_size)
            return image
        
    elif config.data.dataset == 'LSUN':
        dataset_builder = tfds.builder(f'lsun/{config.data.category}', data_dir=config.data.root_path)
        train_split_name = 'train'
        eval_split_name = 'validation'
        # train_split_name = eval_split_name = 'train' ### NOTE: XC change for cats, horses, etc...
        
        if config.data.image_size == 128:
            def resize_op(image):
                image = tf.image.convert_image_dtype(image, tf.float32)
                image = resize_small(image, config.data.image_size)
                iamge = central_crop(image, config.data.image_size)
                return image
        else:
            def resize_op(image):
                image = crop_resize(image, config.data.image_size)
                image = tf.image.convert_image_dtype(image, tf.float32)
                return image
            
    elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
        dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path, data_dir=config.data.root_path)
        train_split_name = eval_split_name = 'train'
        
    else:
        raise NotImplementedError(f"Dataset {config.data.dataset} not yet supportedd.")
    
    # Customize preprocess functions for each dataset
    if config.data.dataset in ['FFHQ', 'CelebAHQ']:
        def preprocess_fn(data):
            sample = tf.io.parse_single_example(
                data,
                features={
                    'shape': tf.io.FixedLenFeature([3], tf.int64),
                    'data': tf.io.FixedLenFeature([], tf.string)
                }
            )
            data = tf.io.decode_raw(sample['data'], tf.uint8)
            data = tf.reshape(data, sample['shape'])
            data = tf.transpose(data, (1, 2, 0))
            image = tf.image.convert_image_dtype(data, tf.float32)
            if config.data.random_flip and not evaluation:
                image = tf.image.random_flip_left_right(image)
            if uniform_dequantization:
                image = (tf.random.uniform(image.shape, dtype=tf.float32) + image * 255.) / 256.
            return dict(image=image, label=None)
    
    else:
        def preprocess_fn(data):
            """
            Basic proprocessing function scales data to [0, 1) and randomly flips.
            """
            image = resize_op(data['image'])
            if config.data.random_flip and not evaluation:
                image = tf.image.random_flip_left_right(image)
            if uniform_dequantization:
                image = (tf.random.uniform(image.shape, dtype=tf.float32) + image * 255.) / 256.
            return dict(image=image, label=data.get('label', None))
    
    def create_dataset(dataset_builder, split):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.experimental_threading.private_threadpool_size = 48
        dataset_options.experimental_threading.max_intra_op_parallelism = 1
        read_config = tfds.ReadConfig(options=dataset_options)
        if isinstance(dataset_builder, tfds.core.DatasetBuilder):
            dataset_builder.download_and_prepare()
            ds = dataset_builder.as_dataset(
                split=split, shuffle_files=True, read_config=read_config
            )
        else:
            ds = dataset_builder.with_options(dataset_options)
        ds = ds.repeat(count=num_epochs)
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds.prefetch(prefetch_size)
    
    train_ds = create_dataset(dataset_builder, train_split_name)
    eval_ds = create_dataset(dataset_builder, eval_split_name)
    
    return train_ds, eval_ds, dataset_builder
    
    
def get_pytorch_dataset(config):
    import pytorch_datasets as tds
    from torchvision.transforms import v2
    
    if config.data.dataset == 'CelebA-HQ-Pytorch':
        transform = v2.Resize(256)
        return tds.celeba_hq_dataset(
            config.training.data_dir,
            config.training.batch_size,
            transform
        ), tds.celeba_hq_dataset(
            config.training.data_dir,
            config.training.batch_size, 
            transform
        )
    elif config.data.dataset == 'AFHQ-CAT-pytorch':
        transform = v2.Resize(256)
        return tds.celeba_hq_dataset(
            config.training.data_dir,
            config.training.batch_size,
            'cat',
            transform
        ), tds.celeba_hq_dataset(
            config.training.data_dir,
            config.training.batch_size, 
            'cat',
            transform
        )
    else:
        assert False, 'Not implemented'