import tensorflow as tf, numpy as np

class MultiLayerPerceptron(object):
    """A base class for full conneted MLP neural network.
    This class is not intended to be instatiated, rather make of SGD or SCSG"""
    
    def __init__(self, lr = 0.05, batch = 50, epochs = 10) :
        self.lr = lr
        self.batch = batch
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.archi = []
        self.logits = None
        self.accuracies = []
        
#         self.metric_fun = tf.metrics.accuracy if metric is None else metric
#         self.metrics = []
        
        self.losses = []
        
        
    def set_weights(self, archi = None, mean=  0,  stddev = 0.05):
        if archi is not None :
            self.archi = archi
        assert self.archi is not None
        
        for i in range(1, len(self.archi)) :
            n1, n2 = archi[i-1], archi[i]
            W = tf.Variable(tf.random_normal([n1, n2], mean = mean, stddev= stddev), name='W%d'%i)
            b = tf.Variable(tf.random_normal([n2]), name='b%d'%i)
            self.weights.append(W)
            self.biases.append(b)
            
    def set_logits(self, x) :
        
        output = tf.add(tf.matmul(x, self.weights[0]), self.biases[0])
        for w, b in zip(self.weights[1:], self.biases[1:]) :
            output = tf.add(tf.matmul(tf.nn.relu(output), w), b)
        
        #self.logits = tf.nn.softmax(output)
        self.logits = output 
                
        
    
    def fit(self, X_train, y_train, X_test, y_test) :
        raise NotImplementedError


class SGD(MultiLayerPerceptron):
    """The Stochastic Gradient Descent Algorithm."""
    def __init__(self, lr, batch, epochs = 10) :
        super().__init__(lr, batch, epochs) 
    
    def fit(self, x_train, y_train, x_test, y_test) :
        
        self.set_weights(self.archi)
        x = tf.placeholder(tf.float32, [None, x_train.shape[1]])
        y = tf.placeholder(tf.float32, [None, y_train.shape[1]])
        
        self.set_logits(x)
        cross_entropy = tf.losses.softmax_cross_entropy(y, self.logits)
        optimiser = tf.train.GradientDescentOptimizer(learning_rate =  self.lr).minimize(cross_entropy)
        accuracy = tf.metrics.accuracy(tf.argmax(y, 1), tf.argmax(self.logits, 1))
        
        
        sel = np.arange(len(y_train))
        
        with tf.Session() as sess :
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            
            for epoch in range(self.epochs):
                np.random.shuffle(sel)
                avg_cost = 0
                for i in range(min(self.batch, len(y_train) ) , len(y_train) +1, self.batch):
                    s = sel[max(0, i - self.batch):i]
                    batch_x, batch_y = x_train[ s ], y_train[ s ]
                    _, c = sess.run([optimiser, cross_entropy], 
                               feed_dict={x: batch_x, y: batch_y})
                    avg_cost += c
                    
#                     if i > 20*self.batch :
#                         break
                
                avg_cost /= (i/self.batch)
                    
                    
                    
                acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
                print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost),  "acuracy: ", acc[1] )




class SCSG(MultiLayerPerceptron):
    """The Stochastic Gradient Descent Algorithm."""
    def __init__(self, lr, batch, epochs = 10) :
        super().__init__(lr, batch, epochs) 

    def get_hyperparams(self, epoch) :
        B_t = 1.5**epoch if self.B is None else (self.B if isinstance(self.B, int) else self.B[epoch] )
        b_t = int(B_t/16) if self.b is None else (self.b if isinstance(self.b, int) else self.b[epoch] )
        eta_t = 0.05 if self.eta is None else (self.eta if isinstance(self.eta, float) else self.eta[epoch])
        return B_t, b_t, eta_t
    
    def fit(self, x_train, y_train, x_test, y_test, eta = None, B=  None, b = None) :
#         eta = 0.05 if eta is None else eta
#         B = 200 if B is None else B
#         b = 10 if b is None else b
        self.B = B
        self.b = b
        self.eta = eta
        
        self.set_weights(self.archi)
        x = tf.placeholder(tf.float32, [None, x_train.shape[1]])
        y = tf.placeholder(tf.float32, [None, y_train.shape[1]])
        
        self.set_logits(x)
        cross_entropy = tf.losses.softmax_cross_entropy(y, self.logits)
        grads =  tf.gradients(cross_entropy, self.weights + self.biases, unconnected_gradients= "zero")
        
#         optimiser = tf.train.GradientDescentOptimizer(learning_rate =  self.lr).minimize(cross_entropy)
        accuracy = tf.metrics.accuracy(tf.argmax(y, 1), tf.argmax(self.logits, 1))
        
        
#         sel = np.arange(len(y_train))
        # res = []
        with tf.Session() as sess :
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sel = np.arange(len(y_train))
            c = 0
            for epoch in range(self.epochs):
                np.random.shuffle(sel)
                B_t, b_t, eta_t = self.get_hyperparams(epoch)
                avg_cost = 0
                for i in range(B_t, len(y_train), B_t):
                    s = sel[i-B_t:B_t]
                    batch_x, batch_y = x_train[ s ], y_train[ s ]

                    g, loss, w, bia = sess.run([grads, cross_entropy, self.weights, self.biases], 
                                                     feed_dict={x: batch_x, y: batch_y})

                    for j in range(len(w)):
                        nu =  g[j]
                        w[j] -= eta_t*nu
                        nu = g[len(w) + j]
                        bia[j] -= eta_t*nu
                    for j in range(len(self.weights)):
                        sess.run(self.weights[j].assign(w[j]))
                        sess.run(self.biases[j].assign(bia[j]))
                    c += 1
                    if i > 10*B :
                        break
                    
                    avg_cost += loss
                    
                avg_cost = avg_cost/c
                    
                acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
                print("Epoch:", (epoch + 1), "cost:", "{:.5f}".format(avg_cost),  "acuracy: ", acc[1] )
                    
