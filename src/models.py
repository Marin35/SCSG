import tensorflow as tf, numpy as np
from time import time

class MultiLayerPerceptron(object):
    """A base class for full conneted MLP neural network.
    This class is not intended to be instantiated, rather make use of SGD or SCSG"""
    
    def __init__(self, lr = 0.05, batch = 50, epochs = 10) :
        self.lr = lr
        self.batch = batch
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.activations = []
        self.archi = []
        self.logits = None
        self.accuracies = []
        self.times = []
        self.losses = []
        
        
    def set_weights(self, archi = None, mean=  0,  stddev = 0.05):
        """Set weights for a multilayer perceptron. 
        The weights are intialised by a standard zero mean normal samples."""
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
        
        z = tf.add(tf.matmul(x, self.weights[0]), self.biases[0])
        for w, b in zip(self.weights[1:], self.biases[1:]) :
            activation = tf.nn.relu(z)
            self.activations.append(activation)
            z = tf.add(tf.matmul(activation, w), b)
            
        activation = tf.nn.softmax(z)
        self.activations.append(activation)
        
        #self.logits = tf.nn.softmax(output)
        self.logits = activation 
                
    def save(self, **kwargs) :
        for key, val in kwargs.items() :
            getattr(self, key).append(val)
    
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
        # accuracy = tf.metrics.accuracy(tf.argmax(y, 1), tf.argmax(self.logits, 1))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), 
                                tf.argmax(self.logits, 1)), tf.float32)) # Prediction accuracy

        
        sel = np.arange(len(y_train))
        np.random.seed(1113)

        with tf.Session() as sess :
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            
            for epoch in range(self.epochs):
                np.random.shuffle(sel)
                k = 0
                for i in range(min(self.batch, len(y_train) ) , len(y_train) +1, self.batch):
                    s = sel[max(0, i - self.batch):i]
                    batch_x, batch_y = x_train[ s ], y_train[ s ]
                    _, c = sess.run([optimiser, cross_entropy], 
                               feed_dict={x: batch_x, y: batch_y})

                    if not k%20 :
                        acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: x_test, y: y_test})
                        self.save(accuracies = acc, losses = loss, times = time())

                    k += 1

                acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: x_test, y: y_test})
                # acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
                print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(loss),  "acuracy: ", acc )




class SCSG(MultiLayerPerceptron):
    """The Stochastic Gradient Descent Algorithm."""
    def __init__(self, epochs = 10) :
        super().__init__( epochs = epochs) 
        self.Ns = []

    def get_hyperparams(self, epoch) :
        B_t = 1.5**epoch if self.B is None else (self.B if isinstance(self.B, int) else self.B[epoch] )
        b_t = int(B_t/16) if self.b is None else (self.b if isinstance(self.b, int) else self.b[epoch] )
        eta_t = 0.05 if self.eta is None else (self.eta if isinstance(self.eta, float) else self.eta[epoch])
        return B_t, b_t, eta_t
    
    def fit(self, x_train, y_train, x_test, y_test, eta = None, B=  None, b = None) :
            self.B = B
            self.b = b
            self.eta = eta
            
            tf.reset_default_graph()  # Restet the graph
            self.set_weights(self.archi)

            x = tf.placeholder(tf.float32, [None, x_train.shape[1]]) # placeholder for x_train
            y = tf.placeholder(tf.float32, [None, y_train.shape[1]])# placeholder for y_train

            self.set_logits(x) # set the ouput layer
            cross_entropy = tf.losses.softmax_cross_entropy(y, self.logits) # loss function

            w_b = self.weights +  self.biases # list of trainable variables
            
            xk = [tf.placeholder(tf.float32, el.shape) for el in w_b] # placeholder for xk update
            x0 = [tf.placeholder(tf.float32, el.shape) for el in w_b] # placeholder for x0 update
            xkOps = [w_b[i].assign(xk[i]) for i in range(len(w_b))]
            x0Ops = [w_b[i].assign(x0[i]) for i in range(len(w_b))]
            
            grads = tf.gradients(cross_entropy, w_b, unconnected_gradients= "zero") # the graident
            
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), 
                                tf.argmax(self.logits, 1)), tf.float32)) # Prediction accuracy

            np.random.seed(113) # The the seet for repoductibility

            with tf.Session() as sess :
                # initialisation
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                
                # Run the given number of epochs
                for epoch in range(self.epochs):
                    
                    B_t, b_t, eta_t = self.get_hyperparams(epoch)

                    sel = np.random.choice(len(y_train), B_t, False )
                    batch_x, batch_y = x_train[ sel ], y_train[ sel ]
                    grads0, w_b_val0 = sess.run([grads, w_b], 
                                    feed_dict={x: batch_x, y: batch_y})
                    
                    N = max(np.random.geometric(B_t/(B_t + b_t)), 2)

                    w_b_val = [ item.copy() for item in w_b_val0]
                    
                    # for g0 in grads0 :
                    #     print("g0:", (g0**2).sum())
                    # print("\n")

                    for j in range(N):
                        sel = np.random.choice(len(y_train), b_t, False )
                        batch_x, batch_y = x_train[ sel ], y_train[ sel ]
                        
                        grads1 = sess.run(grads, feed_dict={x: batch_x, y: batch_y} )
                        updates = {x0[k]: w_b_val0[k] for  k in range(len(w_b)) }
                        updates.update({x: batch_x, y: batch_y})
                        grads2 = sess.run(  [x0Ops, grads], feed_dict= updates)[1]

                        for k in range(len(w_b)):
                            nu = grads1[k] - grads2[k] + grads0[k]
                            w_b_val[k] -= eta_t*nu
                        
                        updates = {xk[k]: w_b_val[k] for  k in range(len(w_b)) }
                        sess.run( xkOps, feed_dict= updates)
                
                
                        acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: x_test, y: y_test})
                        self.save(accuracies = acc, losses = loss, times = time(), Ns = N)  
                        
                    if not epoch%25 :
                        # acc, loss = sess.run([accuracy, cross_entropy], 
                        #                                   feed_dict={x: x_test, y: y_test})
                        print("Epoch:", (epoch + 1), "cost:", "{:.5f}".format(loss), 
                                                                "acuracy: ", acc, "N:", N )