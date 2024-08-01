import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
import copy
def a_norm(Q, K, mask=None):
    m = tf.matmul(Q, tf.transpose(K,perm=[0,2,1]))
    m /= tf.math.sqrt(tf.convert_to_tensor(Q.shape[-1],dtype=tf.float32))
    if mask is not None:
        m = m.masked_fill(mask == 0, -10000)
    return tf.nn.softmax(m,-1)


def time_step_a_norm(Q, K):
    m = tf.matmul(Q, tf.transpose(K,perm=[0,2,1]))
    m /= tf.math.sqrt(tf.convert_to_tensor(Q.shape[-1],dtype=tf.float32))
    return tf.nn.softmax(m,-1)

def attention(Q, K, V, mask=None):
    if mask is not None:
        a = a_norm(Q, K, mask)
    else:
        a = a_norm(Q, K)
    return  tf.matmul(a,  V)


def time_step_attention(Q, K, V):
    a = time_step_a_norm(Q, K)  
    return  tf.matmul(a,  V)


# Query, Key and Value
class Value(Layer):
    def __init__(self, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val
        self.fc1 = tf.keras.layers.Dense(dim_val,activation='relu')
    def __call__(self, x):
        x = self.fc1(x)       
        return x

class Key(Layer):
    def __init__(self, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = tf.keras.layers.Dense(dim_attn,activation='relu')
    
    def __call__(self, x):
        x = self.fc1(x)        
        return x

class Query(Layer):
    def __init__(self, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn        
        self.fc1 = tf.keras.layers.Dense(dim_attn,activation='relu')
        
    def __call__(self, x):        
        x = self.fc1(x)
        return x


class time_step_AttentionBlock(Layer):
    def __init__(self, dim_val, dim_attn,name=None):
        super(time_step_AttentionBlock, self).__init__()
        self.value = Value(dim_val)
        self.key = Key(dim_attn)
        self.query = Query(dim_attn)
        self.name = name
    
    def __call__(self, x, kv = None):
        if(kv is None):
            return time_step_attention(self.query(x), self.key(x), self.value(x))        
        return time_step_attention(self.query(x), self.key(kv), self.value(kv))

class AttentionBlock(Layer):
    def __init__(self, dim_val, dim_attn):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val)  
        self.key = Key(dim_attn)  
        self.query = Query(dim_attn) 
    
    def __call__(self, x, kv = None, mask=None): 
        if(kv is None):
            return attention(self.query(x), self.key(x), self.value(x),mask=mask)        
        return attention(self.query(x), self.key(kv), self.value(kv),mask=mask)

class MultiHeadAttentionBlock(Layer):  
    def __init__(self, dim_val, dim_attn, n_heads): 
        super(MultiHeadAttentionBlock, self).__init__()
        self.n_heads = n_heads
        self.dim_val = dim_val
        self.dim_attn = dim_attn
        self.heads = []
        for _ in range(n_heads):
            self.heads.append(time_step_AttentionBlock(tf.cast(self.dim_val/self.n_heads,dtype=tf.int32),  tf.cast(self.dim_attn/self.n_heads,dtype=tf.int32)))
        self.fc = tf.keras.layers.Dense(dim_val)
        


    def __call__(self, x, kv = None):
        a = []

        x_split = tf.reshape(x, [-1,x.shape[1],tf.cast(self.dim_val/self.n_heads,dtype=tf.int32),self.n_heads])
        i = 0
        for h in self.heads:
            
            a.append(h(x_split[:,:,:,i], kv = kv))
            i = i + 1
        a = tf.concat(a[0:self.n_heads],axis=-1)
        x = self.fc(a)
        return x

class MaskMultiHeadAttentionBlock(Layer):  
    def __init__(self, dim_val, dim_attn, n_heads): 
        super(MaskMultiHeadAttentionBlock, self).__init__()
        self.n_heads = n_heads
        self.dim_val = dim_val
        self.dim_attn = dim_attn
        self.heads = []
        for _ in range(n_heads):
            self.heads.append(AttentionBlock(dim_val,  dim_attn))
        self.fc = tf.keras.layers.Dense(dim_val)


    def __call__(self, x,mask = None):
        a = []
        x_split = tf.reshape(x, [-1,x.shape[1],tf.cast(self.dim_val/self.n_heads,dtype=tf.int32),self.n_heads])
        i = 0
        for h in self.heads:
            
            a.append(h(x_split[:,:,:,i],mask = mask))
            i = i + 1
        a = tf.concat(a[0:self.n_heads],axis=-1)
        x = self.fc(a)
        return x

class TimeStepMultiHeadAttentionBlock(Layer): 
    def __init__(self, dim_val, dim_attn, n_heads): 
        super(TimeStepMultiHeadAttentionBlock, self).__init__()
        self.n_heads = n_heads
        self.dim_val = dim_val
        self.dim_attn = dim_attn
        self.heads = [copy.deepcopy(AttentionBlock(tf.cast(self.dim_val/self.n_heads,dtype=tf.int32), 
                                                   tf.cast(self.dim_val/self.n_heads,dtype=tf.int32))) for _ in range(n_heads)]
        # for i in range(n_heads):
        #     self.heads.append(time_step_AttentionBlock(tf.cast(self.dim_val/self.n_heads,dtype=tf.int32),
        #                                                tf.cast(self.dim_attn/self.n_heads,dtype=tf.int32),
        #                                                name='time_step_AttentionBlock_'+f'{i}'))
        self.fc = tf.keras.layers.Dense(dim_val)

    def __call__(self, x, kv = None):
        a = []
        x_split = tf.reshape(x, [-1,x.shape[1],tf.cast(self.dim_val/self.n_heads,dtype=tf.int32),self.n_heads])
        i = 0
        for h in self.heads:
            a.append(h(x_split[:,:,:,i], kv = kv))
            i = i + 1
        a = tf.concat(a[0:self.n_heads],axis=-1)
        x = self.fc(a)
        return x


#PositionalEncoding (from Transformer)
class PositionalEncoding(Layer):
    def __init__(self, d_model, pe_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe_len = pe_len
        self.pe = tf.zeros([pe_len, d_model],tf.int32)
        # self.pe = tf.stop_gradient(self.pe)
        position = tf.expand_dims(tf.range(0, pe_len, dtype=tf.float32),axis=1)
        
        div_term = tf.math.exp(tf.range(0, d_model, 2,dtype=tf.float32) * (-tf.math.log(10000.0) / d_model))
        self.pe = np.array(self.pe)
        self.pe[:, 0::2] = tf.sin(position * div_term)
        self.pe[:, 1::2] = tf.cos(position * div_term)
        
        self.pe = tf.transpose(tf.expand_dims(self.pe,axis = 0),perm=[1,0,2])
        self.pe = tf.Variable(self.pe,trainable=False)
        
        

    def __call__(self, x):
        x = x + tf.cast(tf.squeeze(self.pe[:x.shape[1], :],axis=1),dtype=tf.float32)
        return x

class Time_step_EncoderLayer(Layer):
    def __init__(self, dim_val, dim_attn, n_heads,dropout=0.1):
        super(Time_step_EncoderLayer, self).__init__()
        self.attn = TimeStepMultiHeadAttentionBlock(dim_val, dim_attn , n_heads)
        self.fc1 = tf.keras.layers.Dense(dim_val)
        self.fc2 = tf.keras.layers.Dense(dim_val)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.elu = tf.keras.layers.ELU()
    
    def __call__(self, x):
        a = self.attn(x)
        a = self.dropout(a)
        x = self.norm1(x + a)
        a = self.fc1(self.elu(self.fc2(x)))
        a = self.dropout(a)
        x = self.norm2(x + a)
        return x



class DecoderLayer(Layer):
    def __init__(self, dim_val, dim_attn, n_heads,dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MaskMultiHeadAttentionBlock(dim_val, dim_attn, n_heads) 
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads) 
        self.fc1 = tf.keras.layers.Dense(dim_val) 
        self.fc2 = tf.keras.layers.Dense(dim_val) 
        self.dropout = tf.keras.layers.Dropout(dropout) 
        self.norm1 = tf.keras.layers.LayerNormalization() 
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()
        self.elu = tf.keras.layers.ELU()

    def __call__(self, x, enc,mask):
        a = self.attn1(x,mask)
        a = self.dropout(a)
        x = self.norm1(a + x)        
        a = self.attn2(x, kv = enc)
        x = self.norm2(a + x)
        a = self.fc1(self.elu(self.fc2(x)))
        a = self.dropout(a)       
        x = self.norm3(x + a)
        return x  



class Trans(Layer):
    def __init__(self,dim_val, dim_attn,n_heads,n_encoder_layers,n_decoder_layers,pe_len):
        super(Trans, self).__init__()
        self.Time_step_EncoderLayer = Time_step_EncoderLayer(dim_val, dim_attn, n_heads)
        self.DecoderLayer = DecoderLayer(dim_val, dim_attn, n_heads)
        self.pe = PositionalEncoding(dim_val, pe_len)
        self.time_encoder = []
        for _ in range(n_encoder_layers):
            self.time_encoder.append(self.Time_step_EncoderLayer)
        self.time_decoder = []
        for _ in range(n_decoder_layers):
            self.time_decoder.append(self.DecoderLayer)
        self.fc = tf.keras.layers.Dense(dim_val)
        self.dense1 = tf.keras.layers.Dense(32,activation='elu')
        self.dense2 = tf.keras.layers.Dense(8,activation='elu')
        self.dense3 = tf.keras.layers.Dense(1,activation='elu')

    def __call__(self, src,trg):
        mask = None
        
        enc_o = self.time_encoder[0](self.pe(self.fc(src)))
        for time_enc in self.time_encoder[1:]:
            enc_o = time_enc(enc_o)
        
        dec_o = self.time_decoder[0](self.pe(self.fc(trg[:,:-1,:])),enc_o,mask)
        for time_dec in self.time_decoder[1:]:
            dec_o = time_dec(dec_o,enc_o,mask)
        dec_o = self.dense3(self.dense2(self.dense1(dec_o)))
        return dec_o[:,:-1,:]


class Trans_encoder(Layer):
    def __init__(self,dim_val, dim_attn,n_heads,n_encoder_layers,pe_len):
        super(Trans_encoder, self).__init__()
        # self.Time_step_EncoderLayer = Time_step_EncoderLayer(dim_val, dim_attn, n_heads)
        self.pe = PositionalEncoding(dim_val, pe_len)
        self.time_encoder = [copy.deepcopy(Time_step_EncoderLayer(dim_val, dim_attn, n_heads)) for _ in range(n_encoder_layers)]
        # self.time_encoder = []
        # for _ in range(n_encoder_layers):
        #     self.time_encoder.append(self.Time_step_EncoderLayer)

        self.fc = tf.keras.layers.Dense(dim_val,activation='elu')

    def __call__(self, x):
        o = self.time_encoder[0](self.pe(self.fc(x)))
        for time_enc in self.time_encoder[1:]:
            o = time_enc(o)
        return o

class Trans_decoder(Layer):
    def __init__(self,dim_val, dim_attn,n_heads,n_decoder_layers,pe_len):
        super(Trans_decoder, self).__init__()
        self.DecoderLayer = DecoderLayer(dim_val, dim_attn, n_heads)
        self.pe = PositionalEncoding(dim_val, pe_len)
        self.time_decoder = []
        for _ in range(n_decoder_layers):
            self.time_decoder.append(self.DecoderLayer)
        self.fc = tf.keras.layers.Dense(dim_val)
        self.out_fc = tf.keras.layers.Dense(dim_val/2,activation='relu')
        self.out_fc2 = tf.keras.layers.Dense(dim_val/4,activation='relu')
        self.out_fc3 = tf.keras.layers.Dense(1, activation='relu')

    def __call__(self, x, enc):
        mask = None
        o = self.time_decoder[0](self.pe(x),enc,mask)
        for time_dec in self.time_decoder[1:]:
            o = time_dec((x),o,mask)
        return o

def transformer(input_size,num_channel,dim_val, dim_attn,n_heads,n_encoder_layers,n_decoder_layers,pe_len):
    inputs = tf.keras.Input((input_size,))
    src = inputs[:,:1024]
    trg = inputs[:,1024:]
    # trg = tf.keras.layers.Embedding(input_dim=1, output_dim=1, mask_zero=True)(trg)
    trg = tf.keras.layers.Masking(mask_value=0)(trg)
    trg = tf.expand_dims(trg,axis=-1)
    src = tf.expand_dims(src,axis=-1)
    outputs = Trans(dim_val, dim_attn,n_heads,n_encoder_layers,n_decoder_layers,pe_len)(src,trg)
    model = tf.keras.Model(inputs=inputs,outputs=outputs)
    return model