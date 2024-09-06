# Хакатон компании Ренью
♻️ Заказчик: Компания Ренью.

📑 Цель проекта: разработать модель для трекинга объектов на конвейерной ленте мусороперерабатывающего завода. 
   - Задача проекта:  Обучить трекер для отслеживания движущихся объектов (пластиковые бутылки разных типов) на ленте конвейера мусороперерабатывающего завода. 

📌 Сроки проекта: 19/08/24 - 09/09/24.

💻 Стек технологий: cv2, ultralytics, YOLO8, BoT-SORT, ByteTrack, SORT, DeepSORT.

## 📝 Описание проекта:
На мусороперерабатывающем заводе над конвейерной лентой установлена камера, которая фиксирует движение пластикового мусора. Данные в потоке передаются детектору и трекеру, которые определяют тип мусора и координаты bounding box.

Необходимо улучшить работу трекера:
- получить более точных координат bounding box;
- обеспечить устойчивость прослеживания объекта без смены ID;

📌 Требования заказчика:
* в течении 2х недель разработать решение для отслеживания объектов на ленте конвейера
* скорость обработки должна быть не более 100мс на кадр
* добиться наилучшего значения метрики MOTA
* подготовить отчет о работе.

⚒️ Работа велась в команде DS:
- [Альбина](https://github.com/usaeva-a)
- [Татьяна](https://github.com/GilevaTanya) 
- [Павел](https://github.com/keyboardnorth) 

## ✅ Результаты
Было протестировано 4 трекера. Наилучшую метрику показал трекер DeepSORT: 

| Трекер | MOTA | Время обработки фрейма (мс) <br> среднее/медиана на 100 фр |
| --- | --- | :-: |
| BotSORT | 0.916468 | 98 / 90 |
| ByteTrack | 0.479714 | 55 / 48 |
| SORT | 0.873508 | 60 / 55 |
| DeepSORT | 0.966587 | 75 / 68 |

Результаты DeepSORT на 9000 фреймах: 
- MOTA = 0.955,
- Время обработки фрейма (среднее/медиана) 105 / 90 мс.

Пример работы трекера DeepSORT:
![](https://github.com/usaeva-a/renew_hackathon/blob/0245a033d1c6dbdf7176b266f00f1dc69487c217/pics/example.gif)

## Структура репозитория
Презентация с результатами хакатона находится [здесь](Renue_results.pptx).
- В папке [test_of_trackers](test_of_trackers) собраны результаты тестирования всех использованных трекеров.
- В папке [app/deepsort](app/deepsort) представлен скрипт для трекера deepsort, показавшего наилучшую метрику.
