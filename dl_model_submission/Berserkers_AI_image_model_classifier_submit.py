# скрипт для предсказания на тестовой выборке.

import os

import torch
import pickle
import numpy as np
import pandas as pd

from PIL import Image

from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder

# предсказания на картинках из папки
images_folder_path = '//content//dataset//health//'
output_csv_filename = 'Berserkers_AI_out.csv'

# путь к сохраненной модели
path_model = 'ResNet34_DGL_AdamW_aug_oversampling_batchsize_16_full_model.pth'

# путь к сохраненному кодировщику классов
path_label_enc = 'label_encoder.pkl'


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

        # для предсказания на папке файлов
        self.prediction_images_values = []
        self.prediction_images_names = []

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

    def get_prediction_on_image_folder(self, images_folder_path):
        """делает предсказание для всех картинок, расположенных в папке images_folder_path,
        моделью нейросети для классифкации картинок, результаты хранит в переменных:
        self.prediction_images_values -  предсказание нейросети
        self.prediction_images_names - название картинки
         """
        # для предсказания на папке файлов
        self.prediction_images_values = []
        self.prediction_images_names = []

        # получаем пути к изображениям
        images_filenames = os.listdir(images_folder_path)

        # предсказываем моделью
        for elem in images_filenames:
            pred = self.get_prediction(images_folder_path + elem)
            pred = 0 if pred == 'health' else 1
            self.prediction_images_values.append(pred)
            self.prediction_images_names.append(elem)

    def get_csv_prediction_on_image_folder(self, output_csv_filename, csv_index_flag=False):
        """создает csv-файл с именем output_csv_filename с предсказаниями модели,
        если csv_index_flag == True в csv-файле колонка-индекс картинок.
        предсказания должны быть предварительно расчитаны методом
        get_prediction_on_image_folder() !
        Нужно для сдачи модели менторам.
        """

        # задаем заголовок csv
        csv_columns = ['disease_flag', 'name']

        # создаем csv
        results_df = pd.DataFrame(list(zip(self.prediction_images_values, self.prediction_images_names)),
                                  columns=csv_columns)
        results_df.to_csv(output_csv_filename, index=csv_index_flag)

    def get_dct_prediction_on_image_folder(self):
        """создает python dict с предсказаниями модели,
        предсказания должны быть предварительно расчитаны методом
        get_prediction_on_image_folder() !
        Нужно для веб-сервиса
        """

        dct = {}
        for i, k in zip(self.prediction_images_names, self.prediction_images_values):
            dct[i] = k

        return dct


# создадим экземпляр класса нашей модели и передадим ей пути к сохраненной нейрости
# и кодировщику классов
web_dl_model = DLModelImageClassifier(path_model, path_label_enc)

# расчитать предсказания
web_dl_model.get_prediction_on_image_folder(images_folder_path)

# опубликовать предсказания в csv-файл
web_dl_model.get_csv_prediction_on_image_folder(output_csv_filename, csv_index_flag=False)