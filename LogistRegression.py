#Data analysis related imports
import pandas as pd
import numpy as np
import random as rnd

#Data vizualization imports
import seaborn as sns
import matplotlib.pyplot as plt


#Machine learning
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

'''
In this Logistic Regression, I will classify Iris flower species
from Iris flower dataset. This dataset has four features and
three labels. Given these four features, we have to predict
flower labels. I will use logistic regression for this
classification problem.
'''
#Load Data
df = pd.read_csv(r'C:\Users\lexdy\Desktop\MachineLearningGoogle\Iris.csv')

#Print some of the data on the Data Frame
print('Some of the data')
print()
print(df.head())
print()


#Print how much data there is in each species
print('Amount of data per species')
print()
print(df['Species'].value_counts())
print()

#Print information of the dataset
print('Some information of the dataset')
print()
print(df.describe())
print()

#Delete the Id collumn because its not needed
df = df.drop('Id', axis=1)

#Convert Species name to a numerical value
#Setosa is 1
#Versicolor = 2
#Virginica = 3
df['Species'] = df['Species'].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[1, 2, 3])
#Now lets print some the updated dataset
print('Fixed dataset')
print()
print(df.head(5))
print()

#Features of the dataset
features = df.loc[:,['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

#Labels of the dataset
labels = df.loc[:, ['Species']]

#Declear the OneHotEncoder from sklearn
oneHot = OneHotEncoder()

#Fit the features to the OneHotEncoder
oneHot.fit(features)

#Transform the Features
features = oneHot.transform(features).toarray()

#Fit the labels to the OneHotEncoder
oneHot.fit(labels)

#Transform the Labels
labels = oneHot.transform(labels).toarray()

print('The features in one-hot format')
print()
print(features)
print()

#Split the out data into the training set and the testing set
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.1, random_state=0)

#Print the shape each set
print('Print the shape of each set')
print()
print('Shape of features_train: ', features_train.shape)
print('Shape of labels-train: ', labels_train.shape)
print('Shape of features_test: ', features_test.shape)
print('Shape of labels_test: ', labels_train.shape)
print()

#Building the model
#Hyperparameters
learning_rate = .0001
num_epochs = 1500
display_step = 1

#Now we start to create each method for the Logistic Regression

with tf.name_scope('Declaring_placeholder'):
    #Features is a placeholder for the iris features. It will be feed later
    features = tf.placeholder(tf.float32, [None, 15])
    #Labels is a placeholder for the iris placeholder. It will be feed later
    labels = tf.placeholder(tf.float32, [None, 3])

with tf.name_scope('Declearing_variables'):
    #w is the weigths. This will be updated during training time
    w = tf.Variable(tf.zeros([15,3]))
    #b is our bias. This will be updated during training time
    b = tf.Variable(tf.zeros([3]))

with tf.name_scope('Declearing_functions'):
    #Our prediction function
    prediction = tf.nn.softmax(tf.add(tf.matmul(features, w), b))

with tf.name_scope('Calculating_cost'):
    #Calculating the cost fucntion
    cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=prediction)
with tf.name_scope('Declaring_Gradient_Descent'):
    #Gradient Descent is gonna be the optimizer thats gonna be used
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('Starting_Session'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            cost_in_each_epoch = 0
            #Training started
            _, c = sess.run([optimizer, cost], feed_dict={features: features_train, labels:labels_train})
            cost_in_each_epoch += c
            #Prints the cost for each epoch
            if(epoch+1)%display_step ==0:
                print("Epoch: {}".format(epoch +1), "cost={}".format(cost_in_each_epoch))

        print('Optimization Finished')

     # Test model
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        # Calculate accuracy for 3000 examples
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({features: features_test, labels: labels_test}))
