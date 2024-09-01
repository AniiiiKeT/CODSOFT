import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2

from Dataset_Processor import DatasetProcessor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Conv2D, MaxPooling2D, Reshape, Input, TimeDistributed
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.regularizers import l2
from tensorflow import keras

from sklearn.model_selection import train_test_split

class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost
    def call(self, y_true, y_pred): 
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

def preprocess_image(image_path, height, width):
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels = 1)
        image = tf.image.resize_with_pad(image, target_height = height, target_width = width)
        image = tf.image.per_image_standardization(image)
        return image
    except tf.errors.InvalidArgumentError:
        return tf.zeros([128, 128, 1])

def preprocess_text(label,max_len,vocabulary):
    layer = tf.keras.layers.StringLookup(vocabulary = vocabulary)
    label = tf.strings.unicode_split(label, input_encoding = "UTF-8")
    label = layer(label)
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values= 99)
    return label

def preprocess(image,label):
    image = preprocess_image(image, img_height, img_width)
    label = preprocess_text(label, max_len, characters)
    return {"image": image, "label": label}

def load_dataset(path):
    images = np.array([])
    labels = np.array([])
    characters = set()  
    max_len = 0
    
    with open(path+ '\\words.txt', 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            if line.startswith('#') or line.strip() == '':
                continue
                
            parts = line.split(" ")
            word_id = parts[0]
            folder1 = word_id.split('-')[0]
            folder2 = folder1 + "-" + word_id.split('-')[1]
            img_path = path + '\\words'+ "\\"+ folder1 + "\\" + folder2+ "\\"+ word_id + '.png'
            if os.path.isfile(img_path) and os.path.getsize(img_path):
                images = np.append(images, img_path)
                label = parts[-1].strip()
                max_len = max(max_len, len(label))
                labels = np.append(labels, label)

                for char in label:
                    characters.add(char)
                    
        return images, labels, max_len, list(characters)

def prepare_dataset(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images,labels)).map(preprocess, num_parallel_calls = tf.data.AUTOTUNE)
    return dataset.batch(32).cache().prefetch(tf.data.AUTOTUNE)

def split_dataset(images, labels):
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.5, random_state=42)
    
    train_set = prepare_dataset(train_images, train_labels)
    val_set = prepare_dataset(val_images, val_labels)
    test_set = prepare_dataset(test_images, test_labels)
    
    return train_set, val_set, test_set

def build_model():
    input_img = tf.keras.Input(shape = (128,128,1), name = 'image')
    labels = tf.keras.layers.Input(name="label", shape=(None,))
    new_shape = ((128 // 4), (128 // 4) * 64)

    x = tf.keras.layers.Conv2D(32,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same")(input_img)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Reshape(target_shape = new_shape)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    x = tf.keras.layers.Dense(len(characters) + 2, activation="softmax", name="dense2")(x)
    output = CTCLayer(name = 'ctc_loss')(labels, x)
    
    model = tf.keras.models.Model(inputs = [input_img, labels], outputs = output)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))
    model.summary()
    return model

def train_model(epochs):
    prediction_model = tf.keras.models.Model(inputs = model.get_layer(name = 'image').input, outputs = model.get_layer(name = 'dense2').output)
    hist = model.fit(
        train_set,
        validation_data=val_set,
        epochs= epochs,
    )
    return hist, prediction_model

def prediction_decode(pred):
    input_len = input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]
    output_text = []
    layer = tf.keras.layers.StringLookup(vocabulary=characters, mask_token=None, invert=True)
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(layer(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text
def inference(test_set, model):
    for batch in test_set.take(1):
        batch_images = batch["image"]
        batch_labels = batch["label"]
        _, ax = plt.subplots(4, 4, figsize=(15, 8))

        preds = model.predict(batch_images)
        pred_texts = prediction_decode(preds)
        for i in range(16):
            img = batch_images[i]
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            title = f"Prediction: {pred_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")
    plt.show()

if __name__ == '__main__':
    
    img_height, img_width = (128,128)
    directory = ("Data")
    path = directory + '\\iam_words'

    images, labels, max_len, characters = load_dataset(path)
    dataset = prepare_dataset(images, labels)
    train_set, val_set, test_set = split_dataset(images,labels)
    model = build_model()
    history, prediction_model = train_model(epochs = 50)
    print(model.evaluate(test_set))

    prediction_model.save('Hand_Written_Text_Generation_Model.h5')
    inference(test_set)



