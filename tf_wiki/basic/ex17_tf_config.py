import tensorflow as tf

print(tf.config.experimental.list_physical_devices())
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(cpus)
tf.config.experimental.set_visible_devices(devices=cpus[0],device_type='CPU')
# 通过 tf.config.set_visible_devices ，可以设置当前程序可见的设备范围（当前程序只会使用自己可见的设备，
# 不可见的设备不会被当前程序使用）

# 使用环境变量 CUDA_VISIBLE_DEVICES 也可以控制程序所使用的 GPU
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"