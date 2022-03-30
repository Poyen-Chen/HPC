import os
import time
import datetime
import tensorflow as tf
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import horovod.tensorflow.keras as hvd

batch_size = 32
epochs = 4

# Initialize Horovod
hvd.init() # HVD

#NUM_PARALLEL_EXEC_UNITS=10
#tf.config.threading.set_intra_op_parallelism_threads(NUM_PARALLEL_EXEC_UNITS)
#tf.config.threading.set_inter_op_parallelism_threads(1)
#tf.config.set_soft_device_placement(True)

#os.environ["OMP_NUM_THREADS"] = "{}".format(NUM_PARALLEL_EXEC_UNITS)
#os.environ["KMP_BLOCKTIME"] = "30"
#os.environ["KMP_SETTINGS"] = "1"
#os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

#physical_devices = tf.config.experimental_list_devices() # Only for TF 2.0
physical_devices = tf.config.list_physical_devices() # For TF >= 2.1
print(physical_devices)

tfds.disable_progress_bar()
tf.enable_v2_behavior()

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs available: ", len(gpus))
local_gpu =  hvd.local_rank() #HVD
if gpus:
    try:
        tf.config.set_visible_devices(gpus[local_gpu], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

ds_train.apply(tf.data.experimental.assert_cardinality(ds_info.splits['train'].num_examples))
ds_test.apply(tf.data.experimental.assert_cardinality(ds_info.splits['test'].num_examples))

num_steps_train = ds_info.splits['train'].num_examples // batch_size
num_steps_test = ds_info.splits['test'].num_examples // batch_size
print("Total number of training steps for training/testing: {}/{}".format(num_steps_train, num_steps_test))

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.shard(hvd.size(), hvd.rank()) # HVD
ds_train = ds_train.cache()
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE).repeat()

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.shard(hvd.size(), hvd.rank()) # HVD
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE).repeat()


def create_model():
    inp = tf.keras.Input(shape=[28, 28, 1])
    flat = tf.keras.layers.Flatten(input_shape=(28, 28, 1))(inp)
    dense1 = tf.keras.layers.Dense(128,activation='relu')(flat)
    dense2 = tf.keras.layers.Dense(10, activation='softmax')(dense1)
    return tf.keras.Model(inp, dense2)

fmodel = create_model()

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0), # HVD
]

opt = tf.optimizers.Adam(0.001 * hvd.size()) # HVD
opt = hvd.DistributedOptimizer(opt) # HVD

# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.
fmodel.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
    experimental_run_tf_function=False # HVD
)

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0: # HVD
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

print("Number of processes: {}".format(hvd.size()))
time_s = time.time()
fmodel.fit(
    ds_train,
    steps_per_epoch=num_steps_train // hvd.size(), # HVD
    epochs=epochs,
    validation_data=ds_test,
    validation_steps=num_steps_test // hvd.size(), # HVD
    callbacks=callbacks,
    verbose=1 if hvd.rank() == 0 else 0)

print("Runtime: {}".format(time.time() - time_s))