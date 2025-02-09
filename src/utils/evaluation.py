import numpy as np
from rouge_score import rouge_scorer
import tensorflow as tf

class SummaryEvaluator:
    def __init__(self):
        """Инициализация оценщика качества резюме."""
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def calculate_rouge_scores(self, reference_summaries, generated_summaries):
        """
        Вычисление метрик ROUGE для сгенерированных резюме.
        
        Args:
            reference_summaries: список эталонных резюме
            generated_summaries: список сгенерированных резюме
            
        Returns:
            словарь с усредненными значениями метрик ROUGE
        """
        scores = {
            'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
            'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
            'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
        }
        
        for ref, gen in zip(reference_summaries, generated_summaries):
            score = self.scorer.score(ref, gen)
            
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                scores[metric]['precision'].append(score[metric].precision)
                scores[metric]['recall'].append(score[metric].recall)
                scores[metric]['fmeasure'].append(score[metric].fmeasure)
        
        # Вычисление средних значений
        avg_scores = {}
        for metric in scores:
            avg_scores[metric] = {
                'precision': np.mean(scores[metric]['precision']),
                'recall': np.mean(scores[metric]['recall']),
                'f1': np.mean(scores[metric]['fmeasure'])
            }
            
        return avg_scores
    
    def evaluate_batch(self, model, preprocessor, input_batch, target_batch, max_length=100):
        """
        Оценка качества генерации резюме для батча данных.
        
        Args:
            model: модель трансформера
            preprocessor: препроцессор текста
            input_batch: батч входных последовательностей
            target_batch: батч целевых последовательностей
            max_length: максимальная длина генерируемого резюме
            
        Returns:
            словарь с метриками качества
        """
        generated_sequences = []
        
        # Генерация резюме для каждого текста в батче
        for inp in input_batch:
            generated_seq = self.generate_summary(model, preprocessor, inp, max_length)
            generated_sequences.append(generated_seq)
        
        # Декодирование последовательностей в текст
        generated_summaries = [preprocessor.decode_sequence(seq) for seq in generated_sequences]
        reference_summaries = [preprocessor.decode_sequence(seq) for seq in target_batch]
        
        # Вычисление метрик ROUGE
        rouge_scores = self.calculate_rouge_scores(reference_summaries, generated_summaries)
        
        return rouge_scores
    
    def generate_summary(self, model, preprocessor, input_seq, max_length):
        """
        Генерация резюме для входного текста.
        
        Args:
            model: модель трансформера
            preprocessor: препроцессор текста
            input_seq: входная последовательность
            max_length: максимальная длина генерируемого резюме
            
        Returns:
            сгенерированная последовательность токенов
        """
        # Добавление размерности батча
        encoder_input = tf.expand_dims(input_seq, 0)
        
        # Начальный токен
        start_token = preprocessor.tokenizer_target.word_index.get(preprocessor.START_TOKEN, 1)
        end_token = preprocessor.tokenizer_target.word_index.get(preprocessor.END_TOKEN, 2)
        
        # Преобразуем в int32
        decoder_input = tf.cast(tf.expand_dims([start_token], 0), dtype=tf.int32)
        
        output_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        output_array = output_array.write(0, start_token)
        
        for i in range(max_length):
            predictions, _ = model([encoder_input, decoder_input], training=False)
            
            # Получение последнего предсказанного токена
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), dtype=tf.int32)
            
            # Если предсказан токен конца последовательности, останавливаемся
            if predicted_id == end_token:
                break
                
            # Сохраняем предсказанный токен
            output_array = output_array.write(i + 1, predicted_id[0, 0])
            
            # Добавление предсказанного токена к выходной последовательности
            decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)
        
        output = output_array.stack()
        return output.numpy() 