#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 06:02:54 2020

@author: ladvien
"""

# Import Tensorflow
import tensorflow as tf

# Import needed tools.
import os
import sys

import matplotlib.pyplot as plt
from random import randint
import numpy as np
from PIL import Image
import wandb

# Import Keras
import tensorflow.keras

image_tools_path = "/home/ladvien/deep_upscale/"
sys.path.append(image_tools_path)

from image_utils import ImageUtils
iu = ImageUtils()


#################################
# TODO: Make experiment folder
#################################
"""
DONE:

TODO:
"""

#################################
# Training Parameters
#################################

root_path               = "/home/ladvien/deep_upscale/"

poor_image_shape        = (128, 128, 1) # This is the shape of the image width, length, colors
rich_image_shape        = (128, 128, 1)

poor_image_size         = (poor_image_shape[0], poor_image_shape[1]) # DOH! image_size is (height, width)
rich_image_size         = (rich_image_shape[0], rich_image_shape[1]) # DOH! image_size is (height, width)
train_test_ratio        = 0.2

# Hyperparameters
batch_size              = 8
epochs                  = 10
steps_per_epoch         = 300
validation_steps        = 50 
optimizer               = 'adam' 
learning_rate           = 0.001
val_save_step_num       = 1
dropout                 = 0.2

path_to_graphs          = f'{root_path}/data/output/logs/'
model_save_dir          = f'{root_path}/data/output/'
train_dir               = f'{root_path}images/'
val_dir                 = f'{root_path}/data/test/'


experiment_settings = {
    "poor_image_shape": poor_image_shape,
    "rich_image_shape": rich_image_shape,
    "poor_image_size": poor_image_size,
    "rich_image_size": rich_image_size,
    "train_test_ratio": train_test_ratio,
    "batch_size":batch_size,
    "epochs":epochs,
    "steps_per_epoch":steps_per_epoch,
    "validation_steps":validation_steps,
    "optimizer":optimizer,
    "learning_rate":learning_rate,
    "val_save_step_num":val_save_step_num,
    "dropout":dropout,
}

wandb.init(config=experiment_settings, project="deep-upscale")

#################################
# Get Train Files
#################################
rich_file_paths = iu.get_image_files_recursively(train_dir + "rich/")


#################################
# Helper functions
#################################

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    

def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history['loss'], label='Train loss')
    ax[0].plot(history.epoch, history.history['val_loss'], label='Validation loss')
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history['acc'], label='Train acc')
    ax[1].plot(history.epoch, history.history['val_acc'], label='Validation acc')
    ax[0].legend()
    ax[1].legend()
    
#################################
# Create needed dirs
#################################
make_dir(path_to_graphs)
make_dir(model_save_dir)

#################################
# Data generators
#################################

# These Keras generators will pull files from disk
# and prepare them for training and validation.

# Determine color depth.
color_mode = ''
if poor_image_shape[2] == 1:
    color_mode = 'grayscale'
elif poor_image_shape[2] == 3:
    color_mode = 'rgb'
elif poor_image_shape[2] == 4:
    color_mode = 'rgba'

print(f'Color mode: {color_mode}')

#################################
# Model Building
#################################

def vae_model(opt, input_shape, dropout = 0.0):
    # Encoder
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (3, 3), input_shape = input_shape, padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    
    # Decoder
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    
    # model.add(tf.keras.layers.Conv2D(256, (3, 3), padding="same"))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.Dropout(dropout))
    # model.add(tf.keras.layers.UpSampling2D((2, 2)))
    
    # model.add(tf.keras.layers.Conv2D(512, (3, 3), padding="same"))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.Dropout(dropout))
    # model.add(tf.keras.layers.UpSampling2D((2, 2)))
    
    model.add(tf.keras.layers.Conv2D(1, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation('sigmoid'))
    
    return model

#################################
# Create model
#################################

def get_optimizer(optimizer, learning_rate = 0.001):
    if optimizer == 'adam':
        return tensorflow.keras.optimizers.Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0., amsgrad = False)
    elif optimizer == 'sgd':
        return tensorflow.keras.optimizers.SGD(lr = learning_rate, momentum = 0.99) 
    elif optimizer == 'adadelta':
        return tensorflow.keras.optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)

selected_optimizer = get_optimizer(optimizer, learning_rate)

model = vae_model(selected_optimizer, poor_image_shape)
model.summary()

model.compile(
    loss = 'binary_crossentropy',
    optimizer = selected_optimizer,
    metrics = ['accuracy']
)
# wandb.watch(model)


best_model_weights = model_save_dir + 'model.h5'


#################################
# Execute Training
#################################

import time
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

train_acc_metric = tf.keras.metrics.BinaryAccuracy()
val_acc_metric = tf.keras.metrics.BinaryAccuracy()

def get_batch(file_paths, batch_size, verbose = 0):
    
    # Batches to return.
    rich_batch = []
    poor_batch = []
    
    # Get random files
    file_nums_to_load = [randint(0, len(file_paths)) - 1 for x in range(0, batch_size)]
    
    # Loop through the batch size, loading files.
    for i in range(0, batch_size):
        
        # Create matching file paths.
        rich_image_file_path = file_paths[file_nums_to_load[i]]
        poor_image_file_path = file_paths[file_nums_to_load[i]].replace("rich", "poor")
        if verbose > 0:
            print(f"Loading rich file: {rich_image_file_path}")
            print(f"Loading poor file: {poor_image_file_path}")
        
        # Load rich image, convert to B&W, and put into training batch.
        rich_image = Image.open(rich_image_file_path).convert("1")
        rich_batch.append(np.array(rich_image, dtype=int))
        
        # Load poor image, convert to B&W, and put into training batch.
        poor_image = Image.open(poor_image_file_path).convert("1")
        poor_batch.append(np.array(poor_image, dtype=int))
        
    return (np.array(poor_batch), np.array(rich_batch))

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step in range(0, steps_per_epoch):
        
        # Load training batch.
        x_batch_train, y_batch_train = get_batch(rich_file_paths, batch_size)
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train.reshape(logits.shape), logits)
            
        grads = tape.gradient(loss_value, model.trainable_weights)
        selected_optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
        # Update training metric.
        train_acc_metric.update_state(y_batch_train.reshape(logits.shape), logits)
    
        # Display metrics at the end of each epoch.
        if step % 10 == 0:
            train_acc = train_acc_metric.result()
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_value}, Accuracy: {float(train_acc)}")
            wandb.log({"loss": loss_value, "accuracy": train_acc})

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()


#################################
# Save Model
#################################
model_json = model.to_json()
with open(model_save_dir + 'model.json', 'w') as json_file:
    json_file.write(model_json)
    
model.save(model_save_dir + 'model.h5')
print('Weights Saved')


#################################
# Test Model
#################################

train_data_path = train_dir + "poor/"
input_path = "/home/ladvien/Desktop/da_output_samps/rare-dragon-58/"

def denoise_random_images_in_folder(path, batch_size, ground_truth_available = True):
    poor_file_paths = iu.get_image_files_recursively(path)
    
    rich_batch = []
    poor_batch = []
    
    random_index = [randint(0, len(poor_file_paths) -1) for x in range(0, batch_size)]
    random_file_paths = [poor_file_paths[i] for i in random_index]
    
    for file_path in random_file_paths:
    
        poor_path = file_path
        rich_path = file_path.replace("poor", "rich")
        
        print(f"Enriching {poor_path}")
        print(f"Comparing {rich_path}")
        
        rich_batch.append(np.array(Image.open(rich_path).convert("1"), dtype=int))
        poor_batch.append(np.array(Image.open(poor_path).convert("1"), dtype=int))    
        

    poor_batch = np.array(poor_batch)
    enriched_batch = model.predict(poor_batch.reshape([batch_size, poor_image_size[0], poor_image_size[0], 1]))
    
    for i in range(0, batch_size):
        plt.title("Poor")
        plt.imshow(poor_batch[i], cmap="gray")    
        plt.show()

        plt.title("Enriched")
        plt.imshow(enriched_batch[i], cmap="gray")    
        plt.show()
        
        if ground_truth_available:
            plt.title("Rich")
            plt.imshow(rich_batch[i], cmap="gray")
            plt.show()
        

denoise_random_images_in_folder(train_data_path, batch_size)

for i in range(10):
    denoise_random_images_in_folder(input_path, batch_size, ground_truth_available = False)

# #################################
# # Clean Images
# #################################

input_path =  "/home/ladvien/deep_arcane/images/1_training/magic_symbols_128x128/train/"
output_path = "/home/ladvien/deep_arcane/images/1_training/magic_symbols_128x128_cleaned/"

files_to_clean = iu.get_image_files_recursively(input_path)

batch_to_clean = []
clean_file_names = []
for i, file in enumerate(files_to_clean, start=1):
    
    batch_to_clean.append(np.array(Image.open(file).convert("1"), dtype=int))
    clean_file_names.append(file)
    
    if i % batch_size == 0:
        
        # Denoise the images.
        cleaned_batch = model.predict(np.array(batch_to_clean).reshape([batch_size, poor_image_size[0], poor_image_size[1], 1]))
        
        # Write them to the new folder.
        for j, clean_file_name in enumerate(clean_file_names):
            
            file_name = clean_file_name.split("/")[-1]
            project_name = clean_file_name.split("/")[-2]
            
            make_dir(output_path + project_name)
            
            # Create the image
            image = Image.fromarray(cleaned_batch[j].reshape(poor_image_size[0], poor_image_size[1]) * 255)
            
            clean_image_path = output_path + project_name + "/" + file_name
            color_img = Image.new("RGB", image.size)
            color_img.paste(image)
            color_img.save(clean_image_path)

        # Reset batch containers.
        batch_to_clean = []
        clean_file_names = []
