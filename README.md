# Object Height Detection

Программа для автоматического определения высоты объектов на изображениях с использованием компьютерного зрения.

## Описание

Алгоритм определяет высоту объектов в пикселях, используя следующие шаги:

1. **Находит контуры объектов** - с помощью методов бинаризации и морфологических операций
2. **Использует cv2.minAreaRect()** - для получения повернутых ограничивающих прямоугольников
3. **Считает реальные высоты** - корректное измерение наклоненных объектов
4. **Вывод результатов** - полную информацию о каждом обнаруженном объекте

## Установка

### Клонирование репозитория
```bash
git clone <repository-url>
cd <repository-folder>
```

### Установка зависимостей
```bash
pip install -r requirements.txt
```

## Использование

### Как скрипт командной строки

#### Одиночное изображение
```bash
python calc_height.py image.jpg
python calc_height.py image.jpg --show
```

#### Пакетная обработка директории
```bash
python calc_height.py --batch ./images
python calc_height.py --batch ./images --show 
```

### Как библиотека в другом коде

#### Базовое использование
```python
import sys
sys.path.append('/path/to/calc_height.py')
from calc_height import find_objects_advanced, find_object_height_advanced

# Полная информация об объектах
objects = find_objects_advanced("image.jpg")
for obj in objects:
    print(f"Высота: {obj['height']:.1f}px, Угол: {obj['angle']:.1f}°")

# Только высоты (совместимость)
heights = find_object_height_advanced("image.jpg")
print(f"Высоты: {heights}")
```

#### Расширенное использование с настройками
```python
from calc_height import find_objects_advanced

# Настройка параметров обработки
objects = find_objects_advanced(
    "image.jpg",
    show=True,              # Визуализация результатов
    target_width=1000,      # Ширина для обработки
    min_area_percent=0.005, # Минимальная площадь объекта
    max_area_percent=0.9    # Максимальная площадь объекта
)

if objects:
    for i, obj in enumerate(objects, 1):
        print(f"Объект {i}:")
        print(f"  Высота: {obj['height']:.1f}px")
        print(f"  Ширина: {obj['width']:.1f}px") 
        print(f"  Центр: ({obj['center'][0]:.1f}, {obj['center'][1]:.1f})")
        print(f"  Угол: {obj['angle']:.1f}°")
        print(f"  Площадь: {obj['area']:.1f}px²")
```

## Примеры вывода

### Консольный вывод
```
Обработка: test01.jpg
Найдено объектов: 2
Объект 1: высота=150.5px, ширина=75.2px, угол=0.0°
Объект 2: высота=200.3px, ширина=100.1px, угол=45.5°
```

### Структура данных объекта
```python
{
    'height': 150.5,           # Реальная высота в пикселях
    'width': 75.2,            # Ширина в пикселях  
    'center': (320.5, 240.3), # Координаты центра
    'angle': 45.5,            # Угол наклона (0-180°)
    'area': 11325.7,          # Площадь в пикселях²
    'perimeter': 451.2,       # Периметр в пикселях
    'aspect_ratio': 2.0,      # Соотношение сторон
    'solidity': 0.95          # Компактность объекта
}
```

## Параметры командной строки

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `image_path` | Путь к изображению | Обязательный |
| `--show` | Показать визуализацию результатов | False |
| `--batch` | Обработать все изображения в директории | - |
| `--width` | Ширина для изменения размера | 800 |
| `--min-area` | Минимальная площадь объекта (%) | 0.003 |
| `--max-area` | Максимальная площадь объекта (%) | 0.95 |

## Параметры функций

### `find_objects_advanced(image_path, show=False, target_width=800, min_area_percent=0.003, max_area_percent=0.95)`

- `image_path` - путь к изображению
- `show` - визуализация промежуточных результатов
- `target_width` - ширина для ресайза (сохранение пропорций)
- `min_area_percent` - минимальная площадь объекта в % от площади изображения
- `max_area_percent` - максимальная площадь объекта в % от площади изображения

### `find_object_height_advanced(image_path, show=False)`
Совместимость с оригинальным кодом - возвращает список высот.

## Особенности алгоритма

- ✅ **Корректное измерение наклоненных объектов** - использование `cv2.minAreaRect()`
- ✅ **Адаптивная обработка** - автоматическая подстройка под разные разрешения
- ✅ **Подавление шумов** - улучшенная бинаризация с очисткой артефактов
- ✅ **Универсальность** - работа с цветными и серыми изображениями
- ✅ **Подробная информация** - полные метрики для каждого объекта

## Поддерживаемые форматы изображений

- JPEG (.jpg, .jpeg)
- PNG (.png) 
- BMP (.bmp)
- TIFF (.tiff)

## Алгоритм работы

1. **Загрузка и предобработка** - улучшение контрастности и ресайз
2. **Бинаризация** - адаптивная пороговая обработка с очисткой шумов
3. **Поиск контуров** - обнаружение и фильтрация объектов
4. **Анализ объектов** - измерение параметров через повернутые прямоугольники
5. **Визуализация** - отображение результатов (опционально)

## Пример интеграции в проект

```python
# measurement_system.py
from calc_height import find_objects_advanced
import json

class ObjectMeasurementSystem:
    def __init__(self):
        self.results = []
    
    def measure_objects(self, image_path):
        """Измерение объектов на изображении"""
        objects = find_objects_advanced(image_path, show=False)
        
        for obj in objects:
            measurement = {
                'height_px': obj['height'],
                'width_px': obj['width'],
                'angle_degrees': obj['angle'],
                'area_px2': obj['area']
            }
            self.results.append(measurement)
        
        return self.results
    
    def save_results(self, output_file):
        """Сохранение результатов в JSON"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

# Использование
system = ObjectMeasurementSystem()
system.measure_objects("product.jpg")
system.save_results("measurements.json")
```

## Требования

- Python 3.6+
- Пакеты из `requirements.txt`
