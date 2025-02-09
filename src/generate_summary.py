import tensorflow as tf
import argparse
from model.transformer import Transformer
from data.preprocessing import TextPreprocessor
from utils.evaluation import SummaryEvaluator

def load_model(model_path, preprocessor, args):
    """Загрузка обученной модели."""
    model = Transformer(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        input_vocab_size=preprocessor.input_vocab_size,
        target_vocab_size=preprocessor.target_vocab_size,
        pe_input=args.max_length_input,
        pe_target=args.max_length_target,
        rate=0.0  # Отключаем dropout при генерации
    )
    
    # Загрузка весов
    model.load_weights(model_path)
    return model

def generate_summary(text, model_path, args):
    """Генерация резюме для входного текста."""
    # Инициализация препроцессора
    preprocessor = TextPreprocessor(
        max_length_input=args.max_length_input,
        max_length_target=args.max_length_target
    )
    
    # Загрузка модели
    model = load_model(model_path, preprocessor, args)
    
    # Подготовка текста
    input_seq = preprocessor.preprocess_data([text], [text])[0]  # Берем только входную последовательность
    
    # Инициализация генератора резюме
    evaluator = SummaryEvaluator()
    
    # Генерация резюме
    generated_seq = evaluator.generate_summary(model, preprocessor, input_seq[0], args.max_length_target)
    
    # Декодирование результата
    summary = preprocessor.decode_sequence(generated_seq)
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Генерация резюме текста')
    
    # Параметры модели (должны совпадать с параметрами обученной модели)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dff', type=int, default=512)
    parser.add_argument('--max_length_input', type=int, default=1000)
    parser.add_argument('--max_length_target', type=int, default=100)
    
    # Путь к модели
    parser.add_argument('--model_path', type=str, required=True,
                        help='Путь к сохраненной модели')
    
    # Входной текст
    parser.add_argument('--text', type=str, required=True,
                        help='Текст для резюмирования')
    
    args = parser.parse_args()
    
    # Генерация резюме
    summary = generate_summary(args.text, args.model_path, args)
    
    print("\nВходной текст:")
    print(args.text)
    print("\nСгенерированное резюме:")
    print(summary)

if __name__ == "__main__":
    main() 