import numpy as np
import tensorflow as tf

#Array of random data input.
x_data = np.random.randn(10000, 6)
#The weight (or labels) of the function
w_real = [0.4, 0.6, 0.2, .5, .10, .12]
#The y-intercept on the function
b_real = -0.1

#This methods are optional, yet i discovered that it
#gives a better output without the noise.
noise = np.random.randn(1,10000)*0.1
y_data = np.matmul(w_real, x_data.T)+b_real+noise

#y_data = np.matmul(w_real, x_data.T)+b_real

#Number if iterations in which the algorithm will train.
num_iters = 10


g = tf.Graph()
wb = []

def TrainIt():
    with g.as_default():
        #Tensor holding the input features values
        x = tf.placeholder(tf.float32, shape=[None, 6], name="features")
        #Tensor holding the label features
        y_true = tf.placeholder(tf.float32, shape=None, name="labels")

        #create the variables that hold the weight, the y-intercept and the labels
        with tf.name_scope('inference') as scope:
            w = tf.Variable([[0,0,0,0,0,0]], dtype=tf.float32, name='W')
            b = tf.Variable(0, dtype=tf.float32, name='b')
            y_pred = tf.matmul(w, tf.transpose(x))+b

        #Holds the mean square loss function
        with tf.name_scope('loss') as scope:
            loss = tf.reduce_mean(tf.square(y_true-y_pred))

        #method that creates the training of the linear regression using gradient descent
        with tf.name_scope('training') as scope:
            #learning rate
            lr = 0.5
            #Optimizer using gradient descent
            optimizer = tf.train.GradientDescentOptimizer(lr)
            #Minimize the gradient descent by the mean square function
            train = optimizer.minimize(loss)

        #Initialize all the variables that were made with tensorflow.
        init = tf.global_variables_initializer()

        #Create the Session for the tensorflow
        with tf.Session() as sess:
            sess.run(init)
            for step in range(num_iters):
                sess.run(train, {x:x_data, y_true:y_data})
                #Every how many iteration does it show the data.
                if(step%2==0):
                    print(step, sess.run([w,b]))
                    wb.append(sess.run([w,b]))




def main():
    TrainIt()

if __name__ == '__main__':
    main()
