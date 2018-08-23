# -*- coding: UTF-8 -*-

import tensorflow as tf

EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
BETA_1 = 0.5

# jug

def discriminator_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(
        64,  # 64 filters,depth 64
        (5, 5),  # 2d size
        padding='same',
        input_shape=(64, 64, 3)
    ))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024))  # full connection
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation("sigmoid"))

    return model

# 生成器
def generator_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(input_dim=100, units=1024))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.Dense(128 * 8 * 8))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.Reshape((8, 8, 128), input_shape=(128 * 8 * 8, )))  # 8 * 8
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))  # 16 * 16
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))  # 32 * 32
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))  # 64 * 64
    model.add(tf.keras.layers.Conv2D(3, (5, 5), padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))

    return model


# 包含生成器与判别器
# input -> xx -> output
def generator_containing_discriminator(generator, discriminator):
    model = tf.keras.models.Sequential()
    model.add(generator)
    discriminator.trainable = False  # 初始时不可训练
    model.add(discriminator)
    return model
