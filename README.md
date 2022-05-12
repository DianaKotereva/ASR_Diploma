# ASR_Diploma

В данном репозитории представлены скрипты для дипломной работы Разработка самообучающейся модели для сегментации речи (Development of a Semi-supervised Speech Segmentation Model)

При написании скриптов были использованы следующие источники:

* https://github.com/felixkreuk/UnsupSeg/blob/master/utils.py
* https://github.com/huggingface
* https://github.com/NVIDIA/NeMo
* https://github.com/lumaku/ctc-segmentation

Структура репозитория:

**Data Preparation** - папка, содержащая тетрадки и скрипты для предобработки данных:
  * Разметка аудио-файлов Golos с помощью алгоритма CTC-Segmentation
  * Удаление тишины из аудио-файлов Golos с помощью silero
  * Распаковка и нарезка Buckeye датасета

**Wav2Vec Segmentation** - папка, содержащая скрипты обучения и прогноза моделей для трех датасетов


Segments:

  * [train](https://drive.google.com/file/d/1hlj8VtiXrTJRmkpRAI3J_cLGvhGgC6Lc/view?usp=sharing)
  * [test](https://drive.google.com/file/d/1P6U83NJAVuK638lfSTD_VuThhHSg6RAd/view?usp=sharing)

Границы Silero Edges:

* [train](https://disk.yandex.ru/d/MNZHudfFkmfGiA)
* [test](https://disk.yandex.ru/d/S8m2f2EhRuQTlA)
 
Сегментация с обрезкой тишины с помощью silero:

* [train](https://disk.yandex.ru/d/Eo5RKDuCzUTM0Q)
* [test](https://disk.yandex.ru/d/AUJ7RwtG7hsSuA)

