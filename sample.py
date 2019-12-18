import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)

# Create some tensors
#a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#c = tf.matmul(a, b)

tf.debugging.set_log_device_placement(True)

# Place tensors on the GPU
with tf.device('/GPU:1'):
  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  c = tf.matmul(a, b)
print(c)

