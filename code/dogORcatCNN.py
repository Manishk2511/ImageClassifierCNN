# importing tensorflow for using CNN and ImageDataGenerator class from keras for preprocessing input data
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# preprocessing training data by augmanting data(for avoiding overfiting) and loading the train data in batche of 32 images

train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_set=train_datagen.flow_from_directory(
    'Section 40 - Convolutional Neural Networks (CNN)/dataset/training_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

# preprocessing test data by augmanting data(for avoiding overfiting) and loading the test data in batches of 32 images
test_datagen=ImageDataGenerator(
        rescale=1./255)
test_set=test_datagen.flow_from_directory(
        'Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set'
    ,target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)


# initializing convolution neural network object
cnn=tf.keras.models.Sequential()

# convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))

# pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

# adding second convolution and pooling layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#flattening the poolmap
cnn.add(tf.keras.layers.Flatten())

# adding full connection layer
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))

#adding ouput layer
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#compiling the cnn (brain)
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#training on train_set
cnn.fit(x=train_set,validation_data=test_set,epochs=2)

# making a single prediction
import numpy as np
from keras.preprocessing import image
test_image=image.load_img('Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set/cats/cat.4937.jpg',target_size=(64,64))
# as input is taken in form of array so convert image to array
test_image=image.img_to_array(test_image)
# as input is taken in batch of 32 so expand the array
test_image=np.expand_dims(test_image,axis=0)
result=cnn.predict(test_image)
train_set.class_indices
if result[0][0]==1:
    prediction='dog'
else :
    prediction='cat'
print(prediction)
