import tensorflow as tf
from .encoder import Encoder
from .decoder import Decoder

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        """
        Полная модель трансформера для задачи резюмирования.
        
        Args:
            num_layers: количество слоев в энкодере и декодере
            d_model: размерность модели
            num_heads: количество головок внимания
            dff: размерность feed-forward сети
            input_vocab_size: размер словаря входных токенов
            target_vocab_size: размер словаря выходных токенов
            pe_input: максимальная длина входной последовательности
            pe_target: максимальная длина выходной последовательности
            rate: dropout rate
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                             input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                             target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def create_masks(self, inp, tar):
        """Создание масок для входных и выходных последовательностей."""
        # Маска для энкодера
        enc_padding_mask = self.create_padding_mask(inp)

        # Маска для декодера
        dec_padding_mask = self.create_padding_mask(inp)

        # Маска для предотвращения доступа к будущим токенам
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def create_padding_mask(self, seq):
        """Создание маски для padding токенов."""
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        """Создание маски для предотвращения доступа к будущим токенам."""
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def call(self, inputs, training):
        inp, tar = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)

        # Энкодер
        enc_output = self.encoder(inp, training, enc_padding_mask)

        # Декодер
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        # Финальный слой
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights 