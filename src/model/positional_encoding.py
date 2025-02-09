import numpy as np
import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        """
        Реализация позиционного кодирования.
        
        Args:
            position: максимальная длина последовательности
            d_model: размерность модели
        """
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, pos, i, d_model):
        """Вычисление углов для позиционного кодирования."""
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def positional_encoding(self, position, d_model):
        """
        Создание матрицы позиционного кодирования.
        
        Args:
            position: максимальная длина последовательности
            d_model: размерность модели
        """
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                   np.arange(d_model)[np.newaxis, :],
                                   d_model)
        
        # Применяем sin к четным индексам
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Применяем cos к нечетным индексам
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        """Добавление позиционного кодирования к входным данным."""
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :] 