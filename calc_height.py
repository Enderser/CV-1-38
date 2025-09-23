import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import sys
import os

def check_grayscale(image):
    """
    Проверяет, является ли изображение серым (1 канал)
    
    Args:
        image: Входное изображение
    
    Returns:
        bool: True если изображение серое
    """
    return len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)

def resize_image(image, target_width=800):
    """
    Изменяет размер изображения с сохранением пропорций
    """
    if image is None:
        return None, 1.0, (0, 0)
        
    original_size = image.shape[:2]
    h, w = original_size
    
    if w == 0:
        return image, 1.0, original_size
    
    scale_factor = target_width / w
    new_width = target_width
    new_height = int(h * scale_factor)
    
    resized = cv2.resize(image, (new_width, new_height))
    return resized, scale_factor, original_size

def enhance_image_contrast(image):
    """
    Улучшение контрастности изображения для лучшего обнаружения объектов
    """
    if check_grayscale(image):
        # Для серых изображений - CLAHE (адаптивное выравнивание гистограммы)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
    else:
        # Для цветных изображений - улучшение в LAB цветовом пространстве
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def preprocess_image(image):
    """
    Универсальная предобработка для разных типов изображений
    """
    if image is None:
        return None
    
    # Улучшение контрастности
    enhanced = enhance_image_contrast(image)
    
    # Конвертация в серое
    if not check_grayscale(enhanced):
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        gray = enhanced
    
    # Несколько методов размытия для разных случаев
    blurred_gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred_median = cv2.medianBlur(gray, 5)
    
    # Выбор лучшего размытия на основе вариации
    gaussian_var = np.var(blurred_gaussian)
    median_var = np.var(blurred_median)
    
    # Меньшая вариация обычно означает лучшее подавление шума
    blurred = blurred_gaussian if gaussian_var < median_var else blurred_median
    
    return blurred

def adaptive_binarization(image, method='combined'):
    """
    Адаптивная бинаризация с несколькими методами
    """
    if image is None:
        return None
    
    methods = {}
    
    # Метод 1: Адаптивная бинаризация Гаусса
    methods['gaussian'] = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Метод 2: Адаптивная бинаризация среднего
    methods['mean'] = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Метод 3: Otsu + Gaussian
    _, otsu_gaussian = cv2.threshold(
        cv2.GaussianBlur(image, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    methods['otsu'] = otsu_gaussian
    
    # Комбинированный метод (лучший из всех)
    if method == 'combined':
        # Оцениваем качество каждого метода по количеству найденных контуров
        best_method = None
        max_contours = 0
        
        for name, binary in methods.items():
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > max_contours:
                max_contours = len(contours)
                best_method = binary
        
        return best_method if best_method is not None else methods['gaussian']
    else:
        return methods.get(method, methods['gaussian'])

def binarize_image(blurred_image):
    """
    Улучшенная бинаризация с морфологическими операциями
    """
    if blurred_image is None:
        return None
    
    # Адаптивная бинаризация
    binary = adaptive_binarization(blurred_image, 'combined')
    
    # Подбор размера ядра в зависимости от размера изображения
    h, w = binary.shape
    kernel_size = max(1, min(h, w) // 300)  # Адаптивный размер ядра
    
    # Морфологические операции для улучшения качества
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Закрытие для заполнения небольших отверстий
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Открытие для удаления небольших шумов
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Дополнительное закрытие для лучшего соединения границ
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def find_contours(binary_image, min_area_percent=0.005, max_area_percent=0.95):
    """
    Улучшенный поиск контуров с фильтрацией по размеру
    """
    if binary_image is None:
        return []
    
    # Поиск контуров с иерархией для лучшего анализа
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return []
    
    # Вычисление площадей для фильтрации
    image_area = binary_image.shape[0] * binary_image.shape[1]
    min_area = int(image_area * min_area_percent)
    max_area = int(image_area * max_area_percent)
    
    filtered_contours = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Фильтрация по площади
        if area < min_area or area > max_area:
            continue
            
        # Фильтрация по компактности (отсеиваем слишком вытянутые объекты)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            if compactness < 0.1:  # Слишком вытянутый объект
                continue
        
        # Проверка на вложенность контуров
        if hierarchy[0][i][3] != -1:  # Если контур вложен в другой
            parent_idx = hierarchy[0][i][3]
            parent_area = cv2.contourArea(contours[parent_idx])
            if area / parent_area > 0.9:  # Если почти такой же как родительский
                continue
        
        filtered_contours.append(contour)
    
    return filtered_contours

def analyze_contour(contour, scale_factor=1.0):
    """
    Анализ контура и извлечение параметров объекта
    """
    if contour is None or len(contour) == 0:
        return None
        
    try:
        # Минимальный ограничивающий прямоугольник
        rect = cv2.minAreaRect(contour)
        center, size, angle = rect
        width, height = size
        
        # Определение реальной высоты и ширины
        real_height = min(width, height)
        real_width = max(width, height)
        
        # Корректировка угла
        if width < height:
            angle += 90
        
        # Масштабирование параметров
        center_original = (center[0] / scale_factor, center[1] / scale_factor)
        width_original = real_width / scale_factor
        height_original = real_height / scale_factor
        
        # Площадь и периметр
        area = cv2.contourArea(contour) / (scale_factor ** 2)
        perimeter = cv2.arcLength(contour, True) / scale_factor
        
        # Дополнительные метрики
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull) / (scale_factor ** 2)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Соотношение сторон
        aspect_ratio = real_width / real_height if real_height > 0 else 0
        
        object_info = {
            'height': height_original,
            'width': width_original,
            'center': center_original,
            'angle': angle % 180,
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'rotated_rect': (center_original, (width_original, height_original), angle % 180)
        }
        
        return object_info
        
    except Exception as e:
        print(f"Ошибка при анализе контура: {e}")
        return None

def draw_rotated_rectangle(image, rotated_rect, color, thickness=2):
    """
    Рисует повернутый прямоугольник на изображении
    """
    if image is None or rotated_rect is None:
        return image
        
    try:
        center, size, angle = rotated_rect
        box_points = cv2.boxPoints(rotated_rect)
        box_points = box_points.astype(np.int32)
        cv2.drawContours(image, [box_points], 0, color, thickness)
    except Exception as e:
        print(f"Ошибка при рисовании прямоугольника: {e}")
    
    return image

def find_objects_advanced(image_path, show=False, target_width=800, 
                         min_area_percent=0.003, max_area_percent=0.95,
                         binarization_method='combined'):
    """
    Универсальная функция для обнаружения объектов на изображениях с разными цветами
    """
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return None
    
    print(f"Обработка: {image_path}")
    print(f"Размер изображения: {image.shape[1]}x{image.shape[0]}")
    
    # Сохранение оригинального изображения
    original_image = image.copy()
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Изменение размера для обработки
    resized_image, scale_factor, original_size = resize_image(image, target_width)
    if resized_image is None:
        print("Ошибка: Не удалось изменить размер изображения")
        return []
    
    # Предобработка
    blurred = preprocess_image(resized_image)
    if blurred is None:
        print("Ошибка: Не удалось выполнить предобработку")
        return []
    
    # Бинаризация
    binary = binarize_image(blurred)
    if binary is None:
        print("Ошибка: Не удалось выполнить бинаризацию")
        return []
    
    # Поиск контуров
    contours = find_contours(binary, min_area_percent, max_area_percent)
    
    print(f"Найдено контуров: {len(contours)}")
    
    if not contours:
        print("Подходящие контуры не найдены")
        # Попробуем альтернативный метод бинаризации
        print("Попытка альтернативной бинаризации...")
        binary_alternative = adaptive_binarization(blurred, 'otsu')
        contours = find_contours(binary_alternative, min_area_percent * 0.5, max_area_percent)
        print(f"Альтернативный метод нашел контуров: {len(contours)}")
    
    # Анализ контуров
    objects_info = []
    for contour in contours:
        object_info = analyze_contour(contour, scale_factor)
        if object_info is not None:
            objects_info.append(object_info)
    
    # Визуализация результатов
    if show and objects_info:
        visualize_results(image_rgb, binary, contours, objects_info, scale_factor)
    elif show:
        print("Нет объектов для отображения")
        # Показать хотя бы бинаризацию
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title('Исходное изображение')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(binary, cmap='gray')
        plt.title('Бинаризация (объекты не найдены)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return objects_info

def visualize_results(original_image, binary_image, contours, objects_info, scale_factor):
    """
    Визуализация результатов обнаружения
    """
    result_image = original_image.copy()
    
    # Цвета для разных объектов
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, (contour, obj_info) in enumerate(zip(contours, objects_info)):
        if obj_info is None:
            continue
            
        color = colors[i % len(colors)]
        
        # Масштабирование контура
        contour_original = (contour / scale_factor).astype(np.int32)
        
        # Рисование контура
        cv2.drawContours(result_image, [contour_original], -1, color, 2)
        
        # Рисование повернутого прямоугольника
        result_image = draw_rotated_rectangle(result_image, obj_info['rotated_rect'], color, 2)
        
        # Добавление информации
        center_x, center_y = map(int, obj_info['center'])
        
        # Основная информация
        text_lines = [
            f'Obj {i+1}',
            f'H: {obj_info["height"]:.1f}px',
            f'W: {obj_info["width"]:.1f}px',
            f'Angle: {obj_info["angle"]:.1f}°'
        ]
        
        # Рисование текста
        for j, line in enumerate(text_lines):
            y_offset = center_y - 20 + j * 15
            cv2.putText(result_image, line, (center_x - 40, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        print(f"Объект {i+1}: высота={obj_info['height']:.1f}px, "
              f"ширина={obj_info['width']:.1f}px, угол={obj_info['angle']:.1f}°, "
              f"площадь={obj_info['area']:.1f}px²")
    
    # Отображение
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Бинаризация')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(result_image)
    plt.title('Обнаруженные объекты')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Совместимость с оригинальным кодом
def find_object_height_advanced(image_path, show=False):
    objects = find_objects_advanced(image_path, show)
    if objects:
        return [obj['height'] for obj in objects]
    return None

def main():
    parser = argparse.ArgumentParser(description='Универсальное обнаружение объектов на изображениях')
    parser.add_argument('image_path', nargs='?', help='Путь к входному изображению')
    parser.add_argument('--show', action='store_true', help='Показать промежуточные результаты')
    parser.add_argument('--batch', help='Обработать все изображения в указанной директории')
    parser.add_argument('--width', type=int, default=800, help='Целевая ширина для изменения размера')
    parser.add_argument('--min-area', type=float, default=0.003, 
                       help='Минимальная площадь контура в процентах')
    parser.add_argument('--max-area', type=float, default=0.95,
                       help='Максимальная площадь контура в процентах')
    
    args = parser.parse_args()
    
    if not args.image_path and not args.batch:
        parser.print_help()
        return
    
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
            
        for image_file in sorted(image_files):
            image_path = os.path.join(args.batch, image_file)
            print(f"\n{'='*50}")
            print(f"Обработка: {image_file}")
            print(f"{'='*50}")
            
            try:
                objects = find_objects_advanced(
                    image_path, args.show, args.width, 
                    args.min_area, args.max_area
                )
                
                if objects:
                    print(f"✓ Найдено объектов: {len(objects)}")
                    for i, obj in enumerate(objects, 1):
                        print(f"  Объект {i}: высота={obj['height']:.1f}px, "
                              f"ширина={obj['width']:.1f}px, угол={obj['angle']:.1f}°")
                else:
                    print("✗ Объекты не обнаружены")
                    
            except Exception as e:
                print(f"❌ Ошибка при обработке {image_file}: {e}")
                continue
                
    else:
        if not os.path.isfile(args.image_path):
            print(f"Ошибка: Файл {args.image_path} не существует")
            sys.exit(1)
            
        objects = find_objects_advanced(
            args.image_path, args.show, args.width, 
            args.min_area, args.max_area
        )
        
        if objects:
            print(f"Найдено объектов: {len(objects)}")
            for i, obj in enumerate(objects, 1):
                print(f"Объект {i}: {obj}")
        else:
            print("Объекты не обнаружены")

if __name__ == "__main__":
    main()