import tensorflow as tf
from tensorflow.keras import Input, layers, models, Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image

def build_model1(input_shape=(32, 32, 3)):
    model1 = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
        layers.BatchNormalization()
    ])

    for _ in range(4):
        model1.add(layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'))
        model1.add(layers.BatchNormalization())

    model1.add(layers.MaxPooling2D(pool_size=(4, 4)))
    model1.add(layers.Flatten())
    model1.add(layers.Dense(128, activation='relu'))
    model1.add(layers.BatchNormalization())
    model1.add(layers.Dense(10, activation='softmax'))

    return model1

def build_model2(input_shape=(32, 32, 3)):

    model2 = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
        layers.BatchNormalization()
    ])

    for _ in range(4):
        model2.add(layers.SeparableConv2D(128, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'))
        model2.add(layers.BatchNormalization())

    model2.add(layers.MaxPooling2D(pool_size=(1, 1)))
    model2.add(layers.Flatten())
    model2.add(layers.Dense(128, activation='relu'))
    model2.add(layers.BatchNormalization())
    model2.add(layers.Dense(10, activation='softmax'))

    return model2

def build_model3(input_shape=(32, 32, 3)):

    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    shortcut = x
    x = layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same')(x)
    x = layers.BatchNormalization()(x)

    shortcut = layers.Conv2D(128, kernel_size=(1, 1), strides=(4, 4), padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    shortcut2 = x
    
    for _ in range(2):
        x = layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    x = layers.Add()([x, shortcut2])
    shortcut3 = x
    for _ in range(2):
        x = layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    x = layers.Add()([x, shortcut3])


    x = layers.MaxPooling2D(pool_size=(4, 4))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create model
    model3 = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model3

def build_model50k(input_shape=(32, 32, 3), dropout_rate=0.2):
    model50k = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.DepthwiseConv2D(kernel_size=(3,3), padding='same'),
        layers.Conv2D(32, kernel_size=(1,1), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.DepthwiseConv2D(kernel_size=(3,3), padding='same'),
        layers.Conv2D(64, kernel_size=(1,1), strides=(4,4), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.DepthwiseConv2D(kernel_size=(3,3), padding='same'),
        layers.Conv2D(96, kernel_size=(1,1), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate)
    ])
    model50k.add(layers.MaxPooling2D(pool_size=(4, 4)))
    model50k.add(layers.Flatten())
    model50k.add(layers.Dense(96, activation='relu'))
    model50k.add(layers.BatchNormalization())
    model50k.add(layers.Dropout(dropout_rate))
    model50k.add(layers.Dense(10, activation='softmax'))
    return model50k

# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
  class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
  train_labels = train_labels.squeeze()
  test_labels = test_labels.squeeze()

  input_shape = train_images.shape[1:]
  train_images = train_images / 255.0
  test_images = test_images / 255.0

  ########################################
  model1 = build_model1()
  model1.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  history = model1.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))
  test_loss, test_acc = model1.evaluate(train_images, train_labels, verbose=2)

#image processing model 1
  image_path = r"C:\Users\FireB\Documents\School Shit\ECGR 4127\2\test_image_plane.png"
  image = Image.open(image_path)
  image = image.resize((32, 32))

  if image.mode != 'RGB':
    image = image.convert('RGB')

  image_array = np.array(image) / 255.0  
  image_array = np.expand_dims(image_array, axis=0)
  predictions = model1.predict(image_array)
  predicted_class = np.argmax(predictions[0])
  print("Predicted class:", predicted_class)

#########################################
  model2 = build_model2()
  model2.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  history2 = model2.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))
  test_loss, test_acc = model2.evaluate(train_images, train_labels, verbose=2)

#image processing model 2
  image_path = r"C:\Users\FireB\Documents\School Shit\ECGR 4127\2\test_image_car.jpg"
  image = Image.open(image_path)
  image = image.resize((32, 32))

  if image.mode != 'RGB':
    image = image.convert('RGB')
  image_array = np.array(image) / 255.0  
  image_array = np.expand_dims(image_array, axis=0)
  predictions = model2.predict(image_array)
  predicted_class = np.argmax(predictions[0])
  print("Predicted class:", predicted_class)

#########################################
  model3 = build_model3()
  model3.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  history = model3.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))
  test_loss, test_acc = model3.evaluate(train_images, train_labels, verbose=2)

#image processing model 3
  image_path = r"C:\Users\FireB\Documents\School Shit\ECGR 4127\2\test_image_dog.jpg"
  image = Image.open(image_path)
  image = image.resize((32, 32))

  if image.mode != 'RGB':
    image = image.convert('RGB')
  image_array = np.array(image) / 255.0 
  image_array = np.expand_dims(image_array, axis=0)
  predictions = model1.predict(image_array)
  predicted_class = np.argmax(predictions[0])

  print("Predicted class:", predicted_class)

  model50k = build_model50k()
  model50k.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  history = model50k.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))
  test_loss, test_acc = model50k.evaluate(train_images, train_labels, verbose=2)

  model50k.save("best_model.h5")

  image_path = r"C:\Users\FireB\Documents\School Shit\ECGR 4127\2\ship.jpg"
  image = Image.open(image_path)
  image = image.resize((32, 32))

  if image.mode != 'RGB':
    image = image.convert('RGB')
  image_array = np.array(image) / 255.0 
  image_array = np.expand_dims(image_array, axis=0)
  predictions = model1.predict(image_array)
  predicted_class = np.argmax(predictions[0])
  print("Predicted class:", predicted_class)

