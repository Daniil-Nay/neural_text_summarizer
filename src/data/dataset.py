import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

def create_sample_dataset():
    """Создание тестового набора данных."""
    data = [
        {
            "text": "The United States and the European Union have reached a major trade agreement after months of intensive negotiations in Brussels. The comprehensive deal will eliminate tariffs on a wide range of goods and services, from agricultural products to digital services. Officials from both sides praised the agreement as a significant step forward in transatlantic relations.",
            "summary": "US and EU reach major trade agreement eliminating tariffs and boosting economic cooperation."
        },
        {
            "text": "Scientists have discovered a remarkable new species of dinosaur in Argentina's Patagonia region. The fossil remains suggest the creature lived approximately 90 million years ago during the Late Cretaceous period. The dinosaur, named Meraxes gigas, was a massive carnivore belonging to the carcharodontosaurid family.",
            "summary": "New dinosaur species Meraxes gigas discovered in Argentina's Patagonia region."
        },
        {
            "text": "A groundbreaking advancement in renewable energy technology has been announced by MIT researchers. The team has developed a revolutionary type of solar cell that can convert sunlight into electricity with unprecedented efficiency. Early laboratory tests show remarkable efficiency rates of up to 40%, significantly higher than current commercial solar panels.",
            "summary": "MIT researchers develop revolutionary solar cell technology achieving 40% efficiency."
        },
        {
            "text": "Apple has unveiled its latest iPhone model at a highly anticipated special event in Cupertino. The new flagship device features a revolutionary AI-powered camera system that uses advanced machine learning algorithms. The company claims the new A18 Bionic processor is twice as fast as its predecessor.",
            "summary": "Apple announces new iPhone with AI-powered camera system and improved processor."
        },
        {
            "text": "Stanford scientists have developed a revolutionary immunotherapy treatment that shows unprecedented success in early clinical trials. The treatment specifically targets aggressive forms of breast cancer by reprogramming the patient's own immune cells. Initial results show an exceptional response rate of over 70% in patients.",
            "summary": "Stanford scientists develop groundbreaking immunotherapy treatment for breast cancer with 70% response rate."
        },
        {
            "text": "NASA's Mars rover Perseverance has made a historic discovery on the red planet. The vehicle has detected complex organic molecules in rock samples collected from an ancient river delta in Jezero Crater. These findings include potential biosignatures that strongly suggest conditions suitable for microbial life.",
            "summary": "NASA's Mars rover discovers organic molecules suggesting possible ancient life on Mars."
        }
    ]
    
    # Создаем больше примеров путем модификации существующих
    extended_data = []
    for item in data:
        # Добавляем оригинальный пример
        extended_data.append(item)
        
        # Создаем первую вариацию
        text1 = item["text"].replace("The", "A").replace("have", "has")
        summary1 = item["summary"].replace("The", "A").replace("develop", "create")
        extended_data.append({
            "text": text1,
            "summary": summary1
        })
        
        # Создаем вторую вариацию
        text2 = item["text"].replace("announced", "revealed").replace("discovered", "found")
        summary2 = item["summary"].replace("announces", "reveals").replace("discovers", "finds")
        extended_data.append({
            "text": text2,
            "summary": summary2
        })
        
        # Создаем третью вариацию
        text3 = item["text"].replace("scientists", "researchers").replace("major", "significant")
        summary3 = item["summary"].replace("scientists", "researchers").replace("major", "significant")
        extended_data.append({
            "text": text3,
            "summary": summary3
        })
    
    return extended_data

def load_dataset(batch_size=64, max_length_input=1000, max_length_target=100, buffer_size=20000):
    """
    Загрузка и подготовка тестового датасета.
    
    Args:
        batch_size: размер батча
        max_length_input: максимальная длина входного текста
        max_length_target: максимальная длина целевого текста
        buffer_size: размер буфера для перемешивания
    
    Returns:
        train_articles, train_summaries, train_dataset, val_dataset, test_dataset
    """
    print("Создание тестового датасета...")
    data = create_sample_dataset()
    
    # Разделяем на тексты и резюме
    texts = [item["text"] for item in data]
    summaries = [item["summary"] for item in data]
    
    # Разделяем на train/val/test
    train_texts, temp_texts, train_summaries, temp_summaries = train_test_split(
        texts, summaries, test_size=0.3, random_state=42
    )
    val_texts, test_texts, val_summaries, test_summaries = train_test_split(
        temp_texts, temp_summaries, test_size=0.5, random_state=42
    )
    
    print(f"Размер датасета:")
    print(f"- Тренировочная выборка: {len(train_texts)} примеров")
    print(f"- Валидационная выборка: {len(val_texts)} примеров")
    print(f"- Тестовая выборка: {len(test_texts)} примеров")
    
    return train_texts, train_summaries, (train_texts, train_summaries), \
           (val_texts, val_summaries), (test_texts, test_summaries) 