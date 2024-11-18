# !pip install anaconda --upgrade
# !pip install tensorflow opencv-python matplotlib

import tensorflow as tf
import os
os.path.join('data', 'happy')
gpus = tf.config.experimental.list_physical_devices('CPU')
gpus
# Avoid OOM (out of memory) errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#Remove Poor Images
import cv2 #OpenCV: Computer Vision Library
import imghdr #Checks file extensions
data_dir = 'data'
os.listdir(os.path.join(data_dir, 'happy')) #goes through every image in subfolder
image_exts = ['jpeg', 'jpg', 'bmp', 'png'] #checks for these specific file types in data
# Define allowed image extensions
image_exts = ['jpeg', 'jpg', 'png', 'bmp']

for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class)
    
    # Check if it's a directory, skip if not
    if not os.path.isdir(class_path):
        continue
    
    for image in os.listdir(class_path):
        image_path = os.path.join(class_path, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))

# Load Dataset
tf.data.Dataset??
import numpy as np
from matplotlib import pyplot as plt
tf.keras.utils.image_dataset_from_directory??
data = tf.keras.utils.image_dataset_from_directory('data', batch_size=8, image_size=(128,128))
data_iterator = data.as_numpy_iterator() #allows access to iterator (useful for large datasets)
batch = data_iterator.next()
batch[0].shape #images represented as numpy arrays
batch[0]
#visualize data and its categories
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])
# PRE-PROCESS DATA
data = data.map(lambda x,y: (x/255, y)) #while loading in batch, scale data
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()
#visualize data and its categories
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])
batch[0].max()
#Split Data into Training/Validation/Test
len(data)
train_size = int(len(data)*.7) #training set is 70% of data
val_size = int(len(data)*.2) #validation set is 20% of data
test_size = int(len(data)*.1) #test set is 10% of data
train = data.take(train_size) #allocating shuffled data
val = data.skip(train_size).take(val_size) 
test = data.skip(train_size+val_size).take(test_size)
# BUILD DEEP-LEARNING Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
model = Sequential()

model.add(Input(shape=(128, 128, 3)))

# Add convolutional layers
model.add(Conv2D(16, (3, 3), activation='relu')) 
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D())

# Add fully connected layers
model.add(Flatten())
model.add(Dense(256, activation='relu')) #Fully-connected Dense Layers
model.add(Dense(1, activation='sigmoid')) #Final, single Dense layer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# train
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
#ideally, loss decreases and accuracy increases consistently
hist.history
#Plot Performance
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()
#if val_loss rises while loss falls, model may be overfitting (apply regularization, add/change data)
#if loss and val_loss don't decrease, need a more sophisticated neural network 
#if val_loss teeters off/up, variance problem (apply regularization)
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()
#High Training Accuracy, Low Val_Accuracy: overfitting (regularization techniques (e.g., L2 regularization, dropout))
#Low Training Accuracy, Low Validation Accuracy: underfitting (Increase model complexity (add more layers/neurons))
#Training Accuracy Increases, Validation Accuracy Stagnates or Decreases: overfitting (Stop training earlier (early stopping) or apply techniques to prevent overfitting like dropout or data augmentation)
# EVALUATE PERFORMANCE
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
import cv2
img = cv2.imread('data/happy/happy_test.jpg')

if img is not None:
    # Convert to RGB (if necessary)
    if img.shape[-1] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the expected input size
    resize = tf.image.resize(img, (256, 256))
    
    # Normalize the image (scale pixel values between 0 and 1)
    resize = resize / 255.0
plt.imshow(resize.numpy())
plt.show()
np.expand_dims(resize, axis=0).shape #resize is the renamed img passed through and resized to fit model
yhat = model.predict(np.expand_dims(resize, 0)) #model expects input to be a batch, input is a single image
yhat
if yhat > 0.5:
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')
#SAVE THE MODEL
from tensorflow.keras.models import load_model
model.save(os.path.join('models','happysadmodel.h5'))
new_model = load_model(os.path.join('models','happysadmodel.h5'))
yhatnew = new_model.predict(np.expand_dims(resize/255, 0))
if yhatnew > 0.5:
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')
