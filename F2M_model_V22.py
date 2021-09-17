# -*- coding:utf-8 -*-
import tensorflow as tf

l1_l2 = tf.keras.regularizers.L1L2(0.00001, 0.000001)
l1 = tf.keras.regularizers.l1(0.00001)

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def attention_residual_block(input, dilation=1, filters=256):
    # Depth wise는 GPU 메로리가 일반 conv보다 많이 필요하기때문에, 쓰는것을 보류
    # 1x1 conv로 해결점을 보자
    h = input

    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[dilation,dilation],[0,0],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 1), padding="valid",
                                kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[0,0],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 3), padding="valid",
                                kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[dilation,dilation],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="valid",
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="valid",
                                kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)

    return h + input

def decode_residual_block(input, dilation=1, filters=256):

    h = input

    x, y = tf.image.image_gradients(input)
    h_attenion_layer = tf.add(tf.abs(x), tf.abs(y))
    h_attenion_layer = tf.reduce_mean(h_attenion_layer, axis=-1, keepdims=True)
    h_attenion_layer = tf.nn.sigmoid(h_attenion_layer)    # attenion map !

    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="valid",
                                kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[dilation,dilation],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="valid",
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[dilation,dilation],[0,0],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 1), padding="valid",
                                kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[0,0],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 3), padding="valid",
                                kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    
    return (h*h_attenion_layer) + input

def F2M_generator(input_shape=(256, 256, 3), de_attention_shape=(256, 256, 1), en_attention_shape=(64, 64, 1)):   # Need to fix parameters!! now!!!!

    h = inputs = tf.keras.Input(input_shape)
    de_attention_layer = tf.keras.Input(de_attention_shape)
    en_attention_layer = tf.keras.Input(en_attention_shape)
    # 여기다가 attention 하나만 추가하자 block 전 layer에 추가하자!
    #1. 지금 weights를 수정한것은 서브컴에 돌리고있음

    #2. 각 attention 기법을 모두 제거, depthwise 1x1을 conv 1x1으로 대체, block에 있는 attention 기법 제거

    #3. 클레스 개수에 따른 weight도 loss에 추가해주자

    #4. discriminator를 두개 써보자 (EnlightenGAN 처럼!)
    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=7,
                               kernel_regularizer=l1_l2, activity_regularizer=l1, name="conv_en_1")(h)
    h = InstanceNormalization()(h)  # [256, 256, 64]
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.DepthwiseConv2D(kernel_size=1, 
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, name="conv_en_2")(h)
    h = InstanceNormalization()(h)  # [256, 256, 64]
    h = tf.keras.layers.ReLU()(h)
    
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1, name="conv_en_3")(h)
    h = InstanceNormalization()(h)  # [128, 128, 128]
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.DepthwiseConv2D(kernel_size=1,
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, name="conv_en_4")(h)
    h = InstanceNormalization()(h)  # [128, 128, 128]
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1, name="conv_en_5")(h)
    h = InstanceNormalization()(h)  # [64, 64, 256]
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.DepthwiseConv2D(kernel_size=1,
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, name="conv_en_6")(h)
    h = InstanceNormalization()(h * en_attention_layer + h)  # [64, 64, 256]

    for i in range(1, 3):
        h = attention_residual_block(h, dilation=i*8, filters=256)

    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same",
                                        kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)  # [128, 128, 128]

    for i in range(1, 4):
        h = decode_residual_block(h, dilation=i * 4, filters=128)

    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same",
                                        kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h * de_attention_layer + h)  # [256, 256, 64]
    h = tf.keras.layers.ReLU()(h)   # 이 마지막 부분에 target 이미지에 대한 attention 을 추가해주자 (low freuency만!)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="valid",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=3, kernel_size=7, strides=1, padding="valid")(h)
    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=[inputs, de_attention_layer, en_attention_layer], outputs=h)

def F2M_discriminator(input_shape=(256, 256, 3)):

    h = inputs = tf.keras.Input(input_shape)
    
    h1 = tf.keras.layers.RandomCrop(128, 128)(h)
    h1 = 1 / (1 + tf.exp(-4.6*h1))
    h1 = tf.reduce_mean(h1, -1, keepdims=True)

    h2 = tf.keras.layers.RandomCrop(64, 64)(h)
    h2 = 1 / (1 + tf.exp(-4.6*h2))
    h2 = tf.reduce_mean(h2, -1, keepdims=True)

    h3 = tf.keras.layers.RandomCrop(32, 32)(h)
    h3 = 1 / (1 + tf.exp(-4.6*h3))
    h3 = tf.reduce_mean(h3, -1, keepdims=True)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h * h1 + h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h) # [128, 128, 64]   이 부분에다 attention map을 추가해보면 ?? 모든 domain 에서 공통적으로 큰 부분만 가지고 온것을 encoder에 접목시키면 괜찮을까?? 즉 공통점만 encoder에 넣는것!

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h * h2 + h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h) # [64, 64, 128]

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h * h3 + h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h) # [32, 32, 256]

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=1, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h) # [32, 32, 512]

    h = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding="same")(h)  # [32, 32, 1]

    return tf.keras.Model(inputs=inputs, outputs=h)

def F2M_discriminator_age(input_shape=(256, 256, 3)):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=1, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding="same")(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)
    
    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(100)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)
