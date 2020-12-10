# -*- coding: utf-8 -*-

"""
Сделал класс-обертку для DL-модели
умеет делать предсказание класса на картинке, ищет GPU, если его нет использует CPU.

Прошу веб специалистов нашей команды посмотреть данный класс в файле "AgroCode pre DLModelImageClassifier.ipynb".
(если вам удобнее просто *py файл то посмотрите "сейчас я его сделаю")

классу  DLModelImageClassifier будут нужны два файла
path_model  - путь до сохраненной нейросети (архитектура и веса одним файлом) файл "ResNet34_DGL_AdamW_aug_oversampling_batchsize_16_full_model.pth"
path_label_enc - путь до сохраненного кодировщика имен классов файл "label_encoder.pkl"

Оба файла в папке "model"
И попробовать сделать тестовый веб-сервер, с сайтом. Чтобы на сайт можно было загрузить одну картинку.
(Картинки есть в папке "pics_example"). И Чтобы сервер напечатал ответ от нейросети.
"""

import torch
import pickle
import numpy as np

from PIL import Image

from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder


class DLModelImageClassifier(object):
    """Класс-обертка над моделью нейросети для классифкации картинок"""

    def __init__(self, path_model, path_label_enc):
        """
        Parameters:

        path_model : str, путь до сохраненной нейросети (архитектура и веса модели PyTorch файлом)
        path_label_enc : str, путь до сохраненного кодировщика имен классов (sklearn.preprocessing.LabelEncoder)
        """

        # загружаем модель
        # на всякий случай грузим модель для cpu
        self.model = torch.load(path_model, map_location=torch.device('cpu'))
        self.model.eval()

        # загружаем кодировщик названий классов
        with open(path_label_enc, 'rb') as f:
            label_enc = pickle.load(f)

        self.label_enc = label_enc

        # определим трансформации картинок для предобработки
        self.our_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Определим устройство, если есть GPU используем его
        # если GPU нет, то CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # перенесем DL-модель на устройство
        self.model.to(self.device)

    def transform_image(self, image_path):
        """делает необходимую трансформацию картинки, расположенной по пути image_path
         для модели нейросети для классифкации картинок"""
        image = Image.open(image_path)

        return self.our_transforms(image).unsqueeze(0)

    def get_prediction(self, image_path):
        """делает предсказание на одной картинке, расположенной по пути image_path
         для модели нейросети для классифкации картинок"""

        # отключаем расчет градиентов, мы только предсказываем
        with torch.no_grad():
            tensor = self.transform_image(image_path=image_path)
            outputs = self.model.forward(tensor.to(self.device))
            _, prediction = torch.max(outputs, 1)
            prediction = prediction.cpu().detach().numpy()

        return self.label_enc.inverse_transform(prediction)[0]


# путь к сохраненной модели
path_model = 'AgroCode_pre_snippets/model/ResNet34_DGL_AdamW_aug_oversampling_batchsize_16_full_model.pth'
# путь к сохраненному кодировщику классов
path_label_enc = 'AgroCode_pre_snippets/model/label_encoder.pkl'

# создадим экземпляр класса нашнй модели и передадим ей пути к сохраненной нейрости
# и кодировщику классов
web_model = DLModelImageClassifier(path_model, path_label_enc)

image_path = 'AgroCode_pre_snippets/pics_example/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG'

# отработал на CPU - сессии Colaba
web_model.device
# вернет device(type='cpu')

# время выполнения засечено в Google colab
# %%time команда Google colab
# сделаем предсказание
web_model.get_prediction(image_path)
# вернет
# CPU times: user 206 ms, sys: 17.5 ms, total: 224 ms
# Wall time: 1.12 s
# Apple___Apple_scab

# отработал на GPU - сессии Colaba
web_model.device
# вернет device(type='cuda')

# время выполнения засечено в Google colab
# %%time команда Google colab
# сделаем предсказание
web_model.get_prediction(image_path)
# вернет
# CPU times: user 21 ms, sys: 6.13 ms, total: 27.2 ms
# Wall time: 681 ms
# Apple___Apple_scab