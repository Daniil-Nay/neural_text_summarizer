import tensorflow as tf
import numpy as np
from model.transformer import Transformer
from model.training import CustomSchedule, train_model
from data.preprocessing import TextPreprocessor
from data.dataset import load_dataset
from utils.evaluation import SummaryEvaluator
import argparse
import os

def main(args):
    # Создание директории для сохранения модели
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Загрузка данных
    print("Загрузка датасета...")
    train_articles, train_summaries, train_data, val_data, test_data = load_dataset(
        batch_size=args.batch_size,
        max_length_input=args.max_length_input,
        max_length_target=args.max_length_target
    )
    
    # Инициализация препроцессора
    print("Подготовка данных...")
    preprocessor = TextPreprocessor(
        max_length_input=args.max_length_input,
        max_length_target=args.max_length_target
    )
    
    # Обучение токенизаторов и подготовка данных
    preprocessor.fit_tokenizers(train_articles, train_summaries)
    input_padded, target_padded = preprocessor.preprocess_data(train_articles, train_summaries)
    
    # Создание датасета
    dataset = preprocessor.create_dataset(
        input_padded, 
        target_padded,
        batch_size=args.batch_size
    )
    
    print(f"Размер словаря входных текстов: {preprocessor.input_vocab_size}")
    print(f"Размер словаря выходных текстов: {preprocessor.target_vocab_size}")
    
    # Инициализация модели
    print("Инициализация модели...")
    model = Transformer(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        input_vocab_size=preprocessor.input_vocab_size,
        target_vocab_size=preprocessor.target_vocab_size,
        pe_input=args.max_length_input,
        pe_target=args.max_length_target,
        rate=args.dropout_rate
    )
    
    # Инициализация оптимизатора
    learning_rate = CustomSchedule(args.d_model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    
    # Обучение модели
    print("Начало обучения...")
    train_model(model, dataset, args.epochs, optimizer)
    
    # Оценка качества
    print("Оценка качества модели...")
    evaluator = SummaryEvaluator()
    
    # Используем тестовые данные
    test_texts, test_summaries = test_data
    
    # Предобработка тестовых данных
    test_input_padded, test_target_padded = preprocessor.preprocess_data(
        test_texts[:5],  # Берем первые 5 примеров для оценки
        test_summaries[:5]
    )
    
    # Вычисление метрик
    scores = evaluator.evaluate_batch(
        model, preprocessor, test_input_padded, test_target_padded
    )
    
    print("\nРезультаты оценки:")
    for metric, values in scores.items():
        print(f"{metric}:")
        for score_type, score in values.items():
            print(f"  {score_type}: {score:.4f}")
    
    # Сохранение модели
    print("\nСохранение модели...")
    model.save_weights(args.model_path)
    print(f"Модель сохранена в {args.model_path}")
    
    # Пример генерации резюме
    print("\nПример генерации резюме:")
    test_text = test_texts[0]
    print("Исходный текст:")
    print(test_text)
    
    # Генерация резюме
    test_input_padded, _ = preprocessor.preprocess_data([test_text], [test_text])
    generated_seq = evaluator.generate_summary(model, preprocessor, test_input_padded[0], args.max_length_target)
    generated_summary = preprocessor.decode_sequence(generated_seq)
    
    print("\nСгенерированное резюме:")
    print(generated_summary)
    print("\nОригинальное резюме:")
    print(test_summaries[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение модели для резюмирования текста')
    
    # Параметры модели
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Количество слоев в энкодере и декодере')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Размерность модели')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Количество головок внимания')
    parser.add_argument('--dff', type=int, default=512,
                        help='Размерность feed-forward сети')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate')
    
    # Параметры данных
    parser.add_argument('--max_length_input', type=int, default=1000,
                        help='Максимальная длина входного текста')
    parser.add_argument('--max_length_target', type=int, default=100,
                        help='Максимальная длина резюме')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Размер батча')
    
    # Параметры обучения
    parser.add_argument('--epochs', type=int, default=20,
                        help='Количество эпох обучения')
    parser.add_argument('--model_path', type=str, default='models/transformer',
                        help='Путь для сохранения модели')
    
    args = parser.parse_args()
    main(args) 