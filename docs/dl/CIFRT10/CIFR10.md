
# Keras- Convolutional Neural Network

-----------

# 3. Image classifier: CIFAR10

##### 3.1 Simple CNN


```python
# Simple CNN model for the CIFAR-10 Dataset
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

```

    Using TensorFlow backend.



```python
K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
```

    Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    170500096/170498071 [==============================] - 15s 0us/step



```python
# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

    WARNING:tensorflow:From /Users/admin/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From /Users/admin/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 32, 32, 32)        896       
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32, 32, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 32, 32, 32)        9248      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 32, 16, 16)        0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 8192)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 512)               4194816   
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                5130      
    =================================================================
    Total params: 4,210,090
    Trainable params: 4,210,090
    Non-trainable params: 0
    _________________________________________________________________
    None
    WARNING:tensorflow:From /Users/admin/anaconda3/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/25
    50000/50000 [==============================] - 164s 3ms/step - loss: 1.6913 - acc: 0.3889 - val_loss: 1.3429 - val_acc: 0.5150
    Epoch 2/25
    50000/50000 [==============================] - 166s 3ms/step - loss: 1.3011 - acc: 0.5330 - val_loss: 1.1536 - val_acc: 0.5947
    Epoch 3/25
    50000/50000 [==============================] - 173s 3ms/step - loss: 1.1191 - acc: 0.6014 - val_loss: 1.0297 - val_acc: 0.6303
    Epoch 4/25
    50000/50000 [==============================] - 175s 3ms/step - loss: 0.9851 - acc: 0.6510 - val_loss: 0.9698 - val_acc: 0.6588
    Epoch 5/25
    48928/50000 [============================>.] - ETA: 3s - loss: 0.8792 - acc: 0.6877

##### 3.3 Plotting


```python
# Plot ad hoc CIFAR10 instances
from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# create a grid of 3x3 images
for i in range(0, 9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(toimage(X_train[i]))
# show the plot
pyplot.show()
```

--------------
