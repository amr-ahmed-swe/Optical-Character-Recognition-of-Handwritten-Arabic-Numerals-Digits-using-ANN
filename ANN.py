import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


"""
There are 60000 photos in the training, and 10000 photos for testing, each one with  28x28 pixel,
with values from 0 to 255.  And there are 60000 labels for the training photos, and 10000 label for testing,
with values  from 0 to 9 
"""  
# read training data set 
train_image = pd.read_csv('/home/amr/Documents/AI/csvTrainImages 60k x 784.csv')
train_label =  pd.read_csv('/home/amr/Documents/AI/csvTrainLabel 60k x 1.csv')

# read testing data set
test_image = pd.read_csv('/home/amr/Documents/AI/csvTestImages 10k x 784.csv')
test_label = pd.read_csv('/home/amr/Documents/AI/csvTestLabel 10k x 1.csv')

"""
The data is between 0 and 255, we will wrap these value to be in 
the range 0.01 to 1 by  multiplying each pixel by 0.99 / 255 and adding 0.01 to the result.
This way, we avoid 0 values  as inputs, which are capable of preventing weight updates.
We are ready now to turn our labelled images into one-hot representations.
Instead of  zeros and one, we create 0.01 and 0.99, which will be better for our calculations 
"""
#pixel values from the range of 0-255 to the range 0-1 preferred for neural network models.

#Scaling data to the range of 0-1 is traditionally referred to as normalization.

#his can be achieved by setting the rescale argument to a ratio by which each pixel can be multiplied to achieve the desired range.

#In this case, the ratio is 1/255 or about 0.0039. 

# normalization
fac = 0.99 / 255
train_image = np.asfarray(train_image) * fac + 0.01
test_image = np.asfarray(test_image) * fac + 0.01
train_label = np.asfarray(train_label)
test_label = np.asfarray(test_label)

# one hot encoding
train_targets = np.array(train_label).astype(np.int)
train_labels_one_hot = np.eye(np.max(train_targets) + 1)[train_targets]
test_targets = np.array(test_label).astype(np.int)
test_labels_one_hot = np.eye(np.max(test_targets) + 1)[test_targets]

train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99
print(train_label[0])          # 1.0
print(train_labels_one_hot[0])  # [0.01 0.99 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01]
"""
activation function :
3.1 Sigmoid for the Hidden Layer 1 
 
3.2 softmax for the Hidden Layer 2  It squish each input 𝑥𝑖 between 0 and 1 
and normalizes the values to give
a  proper probability distribution where the probabilities sum up to one 
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True) 

The sigmoid function always returns a value between 0 and 1.
"""
#Build model

model = Sequential()
model.add( Dense(256,activation='sigmoid',input_dim=784))
model.add( Dense(128,activation='sigmoid'))
model.add( Dense(10,activation='softmax'))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_image, train_label, epochs=3)

test_loss, test_acc = model.evaluate(test_image,  test_label, verbose=2)

print('\nTest accuracy:', test_acc)

#Predictions
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_image)

print(predictions[:4])

print(np.argmax(predictions[:4],axis=1))
#print(test_label[:3])
#a=np.expand_dims(test_label[0], axis=1)
#plt.imshow(a)
def Extract(lst): 
    return [item[0] for item in lst]

lst=test_label[:4]
print(Extract(lst))

for i in range(0,4):
    first_image= test_image[i]
    first_image= np.array(first_image,dtype='float')
    pixels=first_image.reshape((28,28))
    plt.imshow(pixels)
    plt.show()


