import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TextPreprocessor:
    def __init__(self, max_length_input=1000, max_length_target=100):
        """
        Класс для предобработки текстовых данных.
        
        Args:
            max_length_input: максимальная длина входного текста
            max_length_target: максимальная длина целевого текста (резюме)
        """
        self.max_length_input = max_length_input
        self.max_length_target = max_length_target
        
        # Инициализация токенизаторов с добавлением специальных токенов
        self.tokenizer_input = Tokenizer(filters='')
        self.tokenizer_target = Tokenizer(filters='')
        
        # Специальные токены
        self.START_TOKEN = '<start>'
        self.END_TOKEN = '<end>'
        self.PAD_TOKEN = '<pad>'
        
    def fit_tokenizers(self, input_texts, target_texts):
        """
        Обучение токенизаторов на текстах.
        
        Args:
            input_texts: список входных текстов
            target_texts: список целевых текстов (резюме)
        """
        # Добавляем специальные токены к целевым текстам
        target_texts_with_tokens = [
            f"{self.START_TOKEN} {text} {self.END_TOKEN}" for text in target_texts
        ]
        
        # Обучаем токенизаторы
        self.tokenizer_input.fit_on_texts(input_texts)
        self.tokenizer_target.fit_on_texts([self.START_TOKEN, self.END_TOKEN, self.PAD_TOKEN])  # Сначала специальные токены
        self.tokenizer_target.fit_on_texts(target_texts_with_tokens)
        
    def preprocess_data(self, input_texts, target_texts):
        """
        Предобработка текстовых данных.
        
        Args:
            input_texts: список входных текстов
            target_texts: список целевых текстов (резюме)
            
        Returns:
            кортеж (входные последовательности, целевые последовательности)
        """
        # Добавляем специальные токены к целевым текстам
        target_texts_with_tokens = [
            f"{self.START_TOKEN} {text} {self.END_TOKEN}" for text in target_texts
        ]
        
        # Преобразование текстов в последовательности
        input_sequences = self.tokenizer_input.texts_to_sequences(input_texts)
        target_sequences = self.tokenizer_target.texts_to_sequences(target_texts_with_tokens)
        
        # Padding последовательностей
        input_padded = pad_sequences(
            input_sequences, 
            maxlen=self.max_length_input,
            padding='post'
        )
        target_padded = pad_sequences(
            target_sequences,
            maxlen=self.max_length_target,
            padding='post'
        )
        
        return input_padded, target_padded
    
    def create_dataset(self, input_padded, target_padded, batch_size=64, buffer_size=20000):
        """
        Создание tf.data.Dataset для обучения.
        
        Args:
            input_padded: подготовленные входные последовательности
            target_padded: подготовленные целевые последовательности
            batch_size: размер батча
            buffer_size: размер буфера для перемешивания
            
        Returns:
            tf.data.Dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((input_padded, target_padded))
        dataset = dataset.shuffle(buffer_size).batch(batch_size)
        return dataset
    
    @property
    def input_vocab_size(self):
        """Размер словаря входных текстов."""
        return len(self.tokenizer_input.word_index) + 1
    
    @property
    def target_vocab_size(self):
        """Размер словаря целевых текстов."""
        return len(self.tokenizer_target.word_index) + 1
    
    def decode_sequence(self, sequence):
        """
        Декодирование последовательности токенов обратно в текст.
        
        Args:
            sequence: последовательность токенов
            
        Returns:
            декодированный текст
        """
        text = self.tokenizer_target.sequences_to_texts([sequence])[0]
        # Удаляем специальные токены
        text = text.replace(self.START_TOKEN, '').replace(self.END_TOKEN, '').strip()
        return text 