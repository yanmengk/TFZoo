import tensorflow as tf

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])

C = tf.matmul(A, B)
print(C)

# TensorFlow 2.0 版本中，Eager Execution 模式将成为默认模式，无需额外调用 tf.enable_eager_execution() 函数
# （不过若要关闭 Eager Execution，则需调用 tf.compat.v1.disable_eager_execution() 函数）

# 定义一个随机数（标量）
random_float = tf.random.uniform(shape=())
print(random_float)

# 定义一个有2个元素的零向量
zero_vector = tf.zeros(shape=(2))
print(zero_vector)

# 定义两个2×2的常量矩阵
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])
print(A)
print(B)

# 张量的重要属性是其形状、类型和值。可以通过张量的 shape 、 dtype 属性和 numpy() 方法获得。例如：
print(A.dtype)
print(A.shape)
print(A.numpy())

# TensorFlow 的大多数 API 函数会根据输入的值自动推断张量中元素的类型（一般默认为 tf.float32 ）。
# 不过你也可以通过加入 dtype 参数来自行指定类型，
# 例如 zero_vector = tf.zeros(shape=(2), dtype=tf.int32) 将使得张量中的元素类型均为整数。
# 张量的 numpy() 方法是将张量的值转换为一个 NumPy 数组。

C = tf.add(A, B)  # 计算矩阵A和B的和
D = tf.matmul(A, B)  # 计算矩阵A和B的乘积
print(C)
print(D)

# 自动求导机制，求导数
# 计算函数 y(x) = x^2 在 x = 3 时的导数：
x = tf.Variable(initial_value=3.)
# 变量Variable与普通张量的一个重要区别是其默认能够被TensorFlow 的自动求导机制所求导，
# 因此往往被用于定义机器学习模型的参数。

# tf.GradientTape() 是一个自动求导的记录器，在其中的变量和计算步骤都会被自动记录。
with tf.GradientTape() as tape:  # 在tf.GradientTape()的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)  # 计算y关于x的导数
print([y.numpy(), y_grad.numpy()])  # 结果为[9.0, 6.0]

# 对多元函数求偏导数，以及对向量或矩阵求导
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)

with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])
print(L.numpy(), w_grad.numpy(), b_grad.numpy())
