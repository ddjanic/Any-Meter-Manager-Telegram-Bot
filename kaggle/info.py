import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

with tf.device('/cpu:0'):
    ran_matrix = tf.random.uniform(shape='cpu', minval=0, maxval=1)
    d_operation = tf.matmul(ran_matrix, tf.transpose(ran_matrix))
    sum_op = tf.reduce_sum(d_operation)
    start = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    res = session.run(sum_op)
    print(res)
    print("\n" * 6)
    print("Shape:", 'cpu', "Device:", '/cpu:0')
    print("Time done:", datetime.now() - startTime)
    print("\n" * 6)