import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Concatenate, Reshape
from tensorflow.keras.initializers import GlorotUniform


class EScaling(tf.keras.Model):
    def __init__(self, feature_dim, scale, K=1, num_hidden=3, activation="tanh", initializer="glorot_uniform"):
        super(EScaling, self).__init__()
        
        self.title = f"Scaling_{round(feature_dim * scale)}_{num_hidden}_{activation}_{initializer}"
        
        settings = {"activation": activation, "kernel_initializer": initializer}
        self.flat = Flatten()
        self.reshaper = Reshape((round(feature_dim * scale), K))
        self.hidden_layers = []
        for _ in range(num_hidden):
            self.hidden_layers.append(Dense(round(feature_dim * scale) * K, **settings))
            
    def call(self, input_tensor, training=False):
        x = self.flat(input_tensor)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.reshaper(x)
        return x
    

class EMeta(tf.keras.Model):
    def __init__(self, feature_dim, K, C=None):
        super(EMeta, self).__init__()
        
        self.title = f"Meta_{C if C else K}"
        
        self.flat = Flatten()
        self.c = C if C else K
        self.V = tf.Variable(GlorotUniform()(shape=(feature_dim * K, C), dtype=tf.float32), True)
        
    def call(self, input_tensor, training=False):
        x = self.flat(input_tensor)
        x = tf.tile(tf.expand_dims(x, axis=2), (1, 1, self.c))
        x = self.V * x
        return x


class EAutoencoder(tf.keras.Model):
    def __init__(self, feature_dim, squeeze, K=1, activation="tanh", initializer="glorot_uniform"):
        super(EAutoencoder, self).__init__()
        
        self.title = f"Autoencoder_{squeeze}_{activation}_{initializer}"
        
        settings = {"activation": activation, "kernel_initializer": initializer}
        self.k = K
        self.flat = Flatten()
        self.reshaper = Reshape((feature_dim, K))
        self.squeezer = Dense(K * round(feature_dim / squeeze), **settings)
        self.unsqueezer = Dense(K * feature_dim, **settings)
        
    def call(self, input_tensor, training=False):
        x = self.flat(input_tensor)
        x = self.squeezer(x)
        x = self.unsqueezer(x)
        x = self.reshaper(x)
        return x
    
    
class ENet(tf.keras.Model):
    def __init__(self, feature_dim, K, num_hidden=1, activation="tanh", initializer="glorot_uniform"):
        super(ENet, self).__init__()
        
        self.title = f"Net_{num_hidden}_{activation}_{initializer}"
        
        settings = {"activation": activation, "kernel_initializer": initializer} 
        self.k = K
        self.f = feature_dim
        self.flat = Flatten()
        self.concat = Concatenate()
        self.hidden_layers = []
        
        for _ in range(num_hidden):
            self.hidden_layers.append(Dense(feature_dim * (K + 1), **settings))
            
        self.reshape_1 = Reshape((feature_dim, K))
        self.reshape_2 = Reshape((feature_dim, 1))
        
    def call(self, input_tensor, training=False):
        x1 = self.flat(input_tensor[0])
        x2 = self.flat(input_tensor[1])
        
        x = self.concat([x1, x2])
        for layer in self.hidden_layers:
            x = layer(x)
            
        x1 = self.reshape_1(x[:, :self.f * self.k])
        x2 = self.reshape_2(x[:, self.f * self.k:])
        
        return x1, x2


class EWeighting(tf.keras.Model):
    def __init__(self, feature_dim, K=1, activation="sigmoid", initializer="glorot_uniform"):
        super(EWeighting, self).__init__()
        
        self.title = f"Weighting_{activation}_{initializer}"
        
        settings = {"activation": activation, "kernel_initializer": initializer}
        self.k = K
        self.flat = Flatten()
        self.weighter = Dense(feature_dim, **settings)
        self.reshaper = Reshape((feature_dim, K))
        
    def call(self, input_tensor, training=False):
        weights = self.weighter(self.flat(input_tensor))
        if self.k != 1:
            weights = tf.tile(tf.expand_dims(weights, axis=2), (1, 1, self.k))
        else:
            weights = self.reshaper(weights)
        x = input_tensor * weights
        return x