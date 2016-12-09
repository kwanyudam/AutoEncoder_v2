import tensorflow as tf
import numpy as np

def xavier_initialization(input_size, output_size):             #xavier initialization determines initial value from input & output size
    low = -np.sqrt(6.0/(input_size + output_size))
    high = np.sqrt(6.0/(input_size + output_size))
    return tf.random_uniform((input_size, output_size),
        minval=low, maxval=high, dtype=tf.float32)


class NeuralNetwork:
    #Cost Weight for each Position
    def __init__(self, network_arch, ckptname, learning_rate=0.001, decay_rate=0.99, rectifier='relu'):

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.ckptname = ckptname
        
        self.input_shape = network_arch[0]

        self.x = tf.placeholder(tf.float32, [None, network_arch[0]])
        self.y = tf.placeholder(tf.float32, [None, network_arch[0]])
        self.weight =tf.placeholder(tf.float32, [None, 1])

        self.w_recog = [tf.Variable(xavier_initialization(network_arch[i], network_arch[i+1])) for i in range(0, len(network_arch)-1)]
        self.b_recog = [tf.Variable(tf.zeros([network_arch[i]], dtype=tf.float32)) for i in range(0,len(network_arch))]

        self.layers = [tf.mul(self.x, self.weight)]
        for i in range(1, len(network_arch)-1):
            if rectifier=='softplus':
                self.layers.append(tf.nn.softplus(tf.matmul(self.layers[i-1], self.w_recog[i-1])+ self.b_recog[i]))
            elif rectifier=='relu':
                self.layers.append(tf.nn.relu(tf.matmul(self.layers[i-1], self.w_recog[i-1])+ self.b_recog[i]))
            else:
                print "Rectifier\'", rectifier, "\' not recognized"
                raise ValueError(rectifier)

        self.z = tf.matmul(self.layers[-1], self.w_recog[-1]) + self.b_recog[-1]

        #self.w_recon = [tf.Variable(xavier_initialization(network_arch[-i], network_arch[-(i+1)])) for i in range(1, len(network_arch))]
        self.w_recon = [tf.transpose(self.w_recog[len(network_arch)-i-1]) for i in range(1, len(network_arch))]
        self.b_recon = [tf.Variable(tf.zeros([network_arch[-i]], dtype=tf.float32)) for i in range(1,len(network_arch)+1)]

        self.layers_inv = [tf.div(self.z, self.weight)]
        for i in range(1, len(network_arch)-1):
            if rectifier=='softplus':
                self.layers_inv.append(tf.nn.softplus(tf.matmul(self.layers_inv[i-1], self.w_recon[i-1])+self.b_recon[i]))
            elif rectifier=='relu':
                self.layers_inv.append(tf.nn.relu(tf.matmul(self.layers_inv[i-1], self.w_recon[i-1])+self.b_recon[i]))
            else:
                print "Rectifier\'%s\' not recognized" % rectifier
                raise ValueError(rectifier)

        self.result = tf.matmul(self.layers_inv[-1], self.w_recon[-1]) + self.b_recon[-1]

        self.result_loss = tf.reduce_sum(tf.square(self.y - self.result), 1)

        self.cost = tf.reduce_mean(self.result_loss)
	
    	# add tensorboard
        #	cost_summary = tf.scalar_summary("Cost", self.cost)

        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(learning_rate, self.global_step, 1000, decay_rate, staircase=True)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost, global_step=self.global_step)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.cost, global_step=self.global_step)

#	tensorboard
#	self.merged = tf.merge_all_summaries()
#	self.writer = tf.train.SummaryWriter("./cost_logs", self.sess)
        init = tf.initialize_all_variables()

        self.sess.run(init)

        self.saver = tf.train.Saver()

        self.train_count=1

    def train(self, originalBatch, noiseBatch, miss_rate=0.0):
        weight = [1.0 / (1.0-miss_rate)]*len(originalBatch)
        weight = np.array(weight, dtype=float)
        weight = np.reshape(weight, (len(originalBatch), 1))
        
        batch = []
        noise_batch = []
        for i in range(len(originalBatch)):
            batch.append(np.reshape(originalBatch[i], (self.input_shape)))
            noise_batch.append(np.reshape(noiseBatch[i], (self.input_shape)))

        opt,z, cost, gl_step = self.sess.run((self.optimizer,self.z, self.cost, self.global_step),
            feed_dict={self.x:noise_batch, self.y:batch, self.weight:weight})

        #print "Curent step : ", gl_step

        self.train_count = self.train_count + 1

        return cost

    def verify(self, originalBatch, noiseBatch, miss_rate=0):
        weight = [1.0/(1.0-miss_rate)]*len(originalBatch)
        weight = np.array(weight, dtype=float)
        weight = np.reshape(weight, (len(originalBatch), 1))
        
        batch = []
        noise_batch = []
        for i in range(len(originalBatch)):
            batch.append(np.reshape(originalBatch[i], (self.input_shape)))
            noise_batch.append(np.reshape(noiseBatch[i], (self.input_shape)))
        
        
        result, cost = self.sess.run((self.result, self.cost),
            feed_dict={self.x:noise_batch, self.y:batch, self.weight:weight})

        return cost, result

    def reconstruct(self, X, miss_rate=0.0):
        weight = [1.0/(1.0-miss_rate)]
        weight = np.array(weight, dtype=float)
        weight = np.reshape(weight, (1, 1))

        X = np.reshape(X, (1,self.input_shape))
        result = self.sess.run((self.result), feed_dict={self.x:X, self.weight:weight})
        result = np.reshape(result, (15, 3))
        return result

    def save(self):
        self.saver.save(self.sess, self.ckptname)
    def load(self):
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.ckptname)

    def close(self):
        self.sess.close()

    def __call__(self):
        pass
