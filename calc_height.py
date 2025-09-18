import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import sys
import os

def find_object_height_advanced(image_path, show=False):
    """
    Определяет высоту объектов на изображении с использованием компьютерного зрения.
    
    Args:
        image_path (str): Путь к входному изображению
        show (bool): Флаг для отображения промежуточных результатов обработки
    
    Returns:
        list: Список высот обнаруженных объектов в пикселях
    """
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return None
    
    # Конвертация BGR to RGB для правильного отображения в matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Предобработка изображения: преобразование в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Адаптивная бинаризация для выделения объектов
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Морфологические операции для улучшения качества бинаризации
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Закрытие для заполнения небольших отверстий
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)   # Открытие для удаления небольших шумов
    
    # Поиск контуров на бинаризованном изображении
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Контуры не найдены")
        return None
    
    # Фильтрация контуров по минимальной площади
    min_area = 500
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if not filtered_contours:
        print("Подходящие контуры не найдены")
        return None
    
    # Копия изображения для рисования результатов
    result_image = image_rgb.copy()
    
    # Список для хранения высот объектов
    heights = []
    
    # Обработка каждого найденного контура
    for i, contour in enumerate(filtered_contours):
        # Получение ограничивающего прямоугольника для контура
        x, y, w, h = cv2.boundingRect(contour)
        heights.append(h)
        
        # Визуализация результатов, если включен режим показа
        if show:
            cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)  # Рисование контура
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Рисование прямоугольника
            cv2.putText(result_image, f'H: {h}px', (x, y-10),  # Добавление текста с высотой
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            print(f"Объект {i+1}: Высота = {h} пикселей")
    
    # Отображение промежуточных результатов, если включен режим показа
    if show:
        plt.figure(figsize=(15, 5))
        
        # Исходное изображение
        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.title('Исходное изображение')
        plt.axis('off')
        
        # Бинаризованное изображение
        plt.subplot(1, 3, 2)
        plt.imshow(thresh, cmap='gray')
        plt.title('Бинаризация')
        plt.axis('off')
        
        # Результат с обнаруженными объектами
        plt.subplot(1, 3, 3)
        plt.imshow(result_image)
        plt.title('Результат с контурами')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return heights

def main():
    """Основная функция для обработки аргументов командной строки"""
    # Создание парсера аргументов командной строки
    parser = argparse.ArgumentParser(description='Определение высоты объектов на изображении')
    parser.add_argument('image_path', help='Путь к входному изображению')
    parser.add_argument('--show', action='store_true', help='Показать промежуточные результаты обработки')
    parser.add_argument('--batch', help='Обработать все изображения в указанной директории')
    
    args = parser.parse_args()
    
    # Обработка пакетного режима
    if args.batch:
        if not os.path.isdir(args.batch):
            print(f"Ошибка: {args.batch} не является директорией")
            sys.exit(1)
            
        print(f"Обработка изображений в директории: {args.batch}")
        image_files = [f for f in os.listdir(args.batch) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print("В директории не найдено поддерживаемых изображений")
            sys.exit(1)
            
        for image_file in image_files:
            image_path = os.path.join(args.batch, image_file)
            print(f"\nОбработка: {image_file}")
            heights = find_object_height_advanced(image_path, args.show)
            if heights:
                print(f"Найденные высоты: {heights}")
    
    # Обработка одиночного изображения
    else:
        if not os.path.isfile(args.image_path):
            print(f"Ошибка: Файл {args.image_path} не существует")
            sys.exit(1)
            
        heights = find_object_height_advanced(args.image_path, args.show)
        if heights:
            print(f"Найденные высоты: {heights}")

if __name__ == "__main__":
    main()