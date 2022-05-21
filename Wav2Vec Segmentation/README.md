В данной папке находятся скрипты обучения и оценки качества моделей. 

* **transformers_f** - папка, содержащая скрипты библиотеки Transformers, используемая для создания пользовательского класса для модели сегментации речи 
* **modeling_segmentation.py, model_transformers.py, models.py, utils.py** - основные файлы со скриптами 
  * **modeling_segmentation.py** - модель сегментации Wav2Vec2
  * **model_transformers.py** - обертка модели для pytorch_lightning
  * **models.py** - модель сегментации, основанная на работе 2.	Bhati S. et al. Unsupervised Speech Segmentation and Variable Rate Representation Learning using Segmental Contrastive Predictive Coding
  * **utils.py** - алгоритм сэмплирования негативных примеров и функция потерь Contrastive Loss 
* **Transformers_.ipynb** - тетради с обучением моделей
* **Predict_.ipynb** - тетради с оценкой моделей
* **results_.csv** - файлы с оценкой метрик
