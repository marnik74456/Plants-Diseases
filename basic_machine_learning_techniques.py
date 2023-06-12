

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras import  models, layers

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "/content/drive/MyDrive/Datastets/Splitted/train",
        shuffle = True,
        image_size = (IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "/content/drive/MyDrive/Datastets/Splitted/val",
        shuffle = True,
        image_size = (IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "/content/drive/MyDrive/Datastets/Splitted/test",
        shuffle = True,
        image_size = (IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE
)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)


# ANN architecture
ANN =  models.Sequential([
    layers.experimental.preprocessing.Resizing(256, 256),
    layers.experimental.preprocessing.Rescaling(1.0/255),
    layers.Flatten(),
    layers.Dense(units=2048),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(units=1024),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(units=512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    layers.Dense(units=256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(units=128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(units=64, activation='relu'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    layers.Dense(units=33, activation='softmax')
])

ANN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=10**(-7)),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history_logger = CSVLogger('ANN_model.csv', separator=",", append=True)
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

history = ANN.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=val_ds,
                    callbacks=[history_logger, earlyStopping])

ANN.save('ANN_model.keras')


# Basic CNN architecture

CNN =  models.Sequential([
    layers.experimental.preprocessing.Resizing(256, 256),
    layers.experimental.preprocessing.Rescaling(1.0/255),
    layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(2,2)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2), padding='same', strides=(1,1)),

    layers.Conv2D(128, kernel_size=(3,3), padding='same', strides=(1,1)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, kernel_size=(3,3), padding='same', strides=(2,2)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2), padding='same', strides=(1,1)),

    layers.Conv2D(256, kernel_size=(3,3), padding='same', strides=(1,1)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(256, kernel_size=(2,2), padding='same', strides=(2,2)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2), padding='same', strides=(1,1)),

    layers.Conv2D(512, kernel_size=(3,3), padding='same', strides=(1,1)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(512, kernel_size=(2,2), padding='same', strides=(2,2)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2), padding='same', strides=(1,1)),

    layers.Flatten(),
    layers.Dense(units=1024),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    layers.Dense(units=512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    layers.Dense(units=128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(units=33, activation='softmax')
])

CNN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=10**(-7)),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history_logger = CSVLogger('ANN_model.csv', separator=",", append=True)
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

history = CNN.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=val_ds,
                    callbacks=[history_logger, earlyStopping])

CNN.save('CNN_model.keras')

