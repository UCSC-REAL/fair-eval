import tensorflow as tf
import tensorflow_datasets as tfds


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def load_celeba_dataset(args, shuffle_files=False, batch_size=128):
  ds_train, ds_test = tfds.load(name='celeb_a', split=['train', 'test'], data_dir=args.data_dir,
      batch_size=batch_size, download=True, shuffle_files=shuffle_files)
  return ds_train, ds_test
  
