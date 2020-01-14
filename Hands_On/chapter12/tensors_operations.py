import tensorflow as tf

t = tf.constant([[1.,2.,3.],[4.,5.,6.]])
print(t)
# tf.Tensor(
# [[1. 2. 3.]
#  [4. 5. 6.]], shape=(2, 3), dtype=float32)
print(t.shape) # (2, 3)
print(t.dtype) # <dtype: 'float32'>

# 对tensor进行索引
print(t[:,1:])
# tf.Tensor(
# [[2. 3.]
#  [5. 6.]], shape=(2, 2), dtype=float32)

print(t[:,1])
# 注意：此处结果为 tf.Tensor([2. 5.], shape=(2,), dtype=float32)

print(t[:,1,tf.newaxis])
# tf.Tensor(
# [[2.]
#  [5.]], shape=(2, 1), dtype=float32)

print(t+10) # 等同于 tf.add(t,10)

print(tf.square(t))

print(t @ tf.transpose(t)) # @表示矩阵乘法，等同于tf.matmul


print(tf.constant(42))
# tf.Tensor(42, shape=(), dtype=int32) 标量的shape=()

# Variables 变量
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
