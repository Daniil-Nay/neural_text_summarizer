import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        """
        Реализация Multi-Head Attention механизма.
        
        Args:
            d_model: размерность модели
            num_heads: количество головок внимания
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        # Создаем слои для Query, Key, Value
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Разделение последнего измерения на (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        """Вычисление attention весов.
        
        Args:
            q: query shape == (..., seq_len_q, depth)
            k: key shape == (..., seq_len_k, depth)
            v: value shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k)
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Масштабирование
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Применение маски
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Софтмакс по последней оси
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        output = tf.matmul(attention_weights, v)
        return output, attention_weights
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        # Линейные преобразования
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # Разделение на головки
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Вычисление attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        # Возвращаем форму к исходной
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))
        
        # Финальное линейное преобразование
        output = self.dense(concat_attention)
        
        return output, attention_weights 