import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

Dataset =(r'C:\Users\lexdy\Desktop\MachineLearningGoogle\DataSets\fire_theft.xls')

#Read the data from the .xls file
book = xlrd.open_workbook(Dataset, encoding_override = "utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range (1 , sheet.nrows)])
nsample = sheet.nrows -1

#Create the placeholders for the X inputs (numbers of fires) and lables Y (numbers of theft)
X = tf.placeholder(tf.float32, name = "X")
Y = tf.placeholder(tf.float32, name = "Y")

#Create the weight and the bias
w = tf.Variable(0.0, name = "weigths")
b = tf.Variable(0.0, name = 'bias')

#Construct the model to predict Y (number of theft) from the number of fire
Y_predict = X*w + b

#Use the square error function as the loss function
loss = tf.square(Y - Y_predict, name = "loss")

#Set the Gradient Descent as the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = .001).minimize(loss)

with tf.Session() as sess:
    #Initialize the necesarry variables (this case would be weigth and bias)
    sess.run(tf.global_variables_initializer())

    #Train the model
    for i in range(100): #run for 1000 epochs
        for x, y in data:
            #Session runs train_op to minimize loss
            sess.run(optimizer, feed_dict={X:x, Y:y})
            w_value, b_value = sess.run([w,b])
            print("epochs: ", i , "W: " ,w_value, "B:" , b_value, "loss:", loss)
