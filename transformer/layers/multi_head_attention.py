import tensorflow as tf
from tensorflow.keras.layers import Dense


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model = 512, h = 6):
        """
            Parameters
            ----------
            d_model: int
                - the dimension of linear projection of q, k and v
            h: int
                - the number of heads


        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        # Num of heads
        self.h = h
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.wo = Dense(d_model)


    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
            Calculate Attention score
            Parameters
            ----------
            q: tensor
                - query
                - shape: (..., q_length, d_k)
            k: tensor
                - key
                - shape: (..., k_lengh, d_k)
            v: tensor
                - value
                - shape: (..., v_length, d_v)
            k_lengh = v_length

            Returns
            ----------
            attention_weights: tensor 
                - Attention Scores between Query and Key
                - shape: (..., q_length, k_lengh)
            out: tensor
                - Attention Weights on Value
                - shape: (..., q_length, k_lengh)
        """

        dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)

        attention_scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk) # (..., q_length, d_k) . (..., d_k, k_lengh) = (..., q_length, k_lengh)

        if mask:
            attention_scores += (mask * -1e30)

        attention_weights =  tf.nn.softmax(attention_scores, axis=-1) 

        out = tf.matmul(attention_weights, v) # (..., q_length, k_lengh) . (k_lengh, d_v) = (..., q_length, d_v)

        return out, attention_weights

    def splitting_head(self, x):
        """
            Splitting item to heads
            Parameters
            ----------
            x: tensor
                - query/key/value
                - shape: (..., length, d_model)
            Returns
            ----------
            xs: tensor
                - splitted heads
                - shape: (..., h, length, hd_v)

        """
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1] 
        d_model = tf.shape(x)[2] 

        # assert d_model % self.h == 0
        
        hd_v = d_model // self.h

        x = tf.reshape(x, (batch_size, length, self.h, hd_v)) 

        xs = tf.transpose(x, [0, 2, 1, 3]) # (..., h, length, hd_v)
        
        return xs
        


    def call(self, q, k, v, mask=None):
        """
            Calculate Attention score
            Parameters
            ----------
            q: tensor
                - query
                - shape: (..., q_length, d_k)
            k: tensor
                - key
                - shape: (..., k_lengh, d_k)
            v: tensor
                - value
                - shape: (..., v_length, d_v)
            k_lengh = v_length

            Returns
            ----------
            attention_weights: tensor 
                - Attention Scores between Query and Key
                - shape: (..., q_length, k_lengh)
            out: tensor
                - Attention Weights on Value
                - shape: (..., q_length, k_lengh)
        """

        batch_size = tf.shape(q)[0]
        qw = self.wq(q) # (..., q_length, d_model)
        kw = self.wk(k) # (..., k_lengh, d_model)
        vw = self.wv(v) # (..., k_lengh, d_model)

        # Splitting Head

        heads_qw = self.splitting_head(qw) # (..., h, q_length, hd_v)
        heads_kw = self.splitting_head(kw) # (..., h, k_lengh, hd_v)
        heads_vw = self.splitting_head(vw) # (..., h, k_lengh, hd_v)

        # Do Attention
        # attention_weights shape: # (..., h, q_length, k_lengh)
        # out shape: # (..., h, q_length, hd_v)

        out, attention_weights = self.scaled_dot_product_attention(heads_qw, heads_kw, heads_vw)

        # Transpose out back to # (..., q_length, d_model)

        out = tf.transpose(out, [0, 2, 1, 3])  # (..., q_length, h, hd_v)

        out = tf.reshape(out, (batch_size, tf.shape(qw)[1], self.d_model)) # (..., q_length, d_model)

        final = self.wo(out) # (..., q_length, d_model)

        return final, attention_weights
