import pandas as pd
from catboost import CatBoostClassifier


# Класс предобработки данных
class Preprocess:
    def __init__(self):
        # Список названий библиотек, которые используем как признаки
        self.libs_features = []

    def fit(self, df):
        df = df.copy()
        # Создадим список названий вместо записи в строку через запятую
        df['libs'] = df['libs'].apply(str.split, args=',')
        # Список всех встречающихся библиотек запишем в наш атрибут
        self.libs_features = list(set(df['libs'].sum()))

    def transform(self, df):
        df = df.copy()
        df['libs'] = df['libs'].apply(str.split, args=',')
        # Добавим в датафрейм столбцы с новыми признаками
        features_df = pd.DataFrame(data=0,
                                   index=df.index,
                                   columns=self.libs_features)
        df = pd.concat([df, features_df], axis=1)

        # Проведем One hot encoding наших признаков
        for idx in df.index:
            libs = df.iloc[idx]['libs']
            for lib in libs:
                df.at[idx, lib] = 1

        # Добавим еще один признак - количество используемых библиотек
        # Так как мы используем деревья решений, то модель не "выучит"
        # несуществующую линейную зависимость и интерпретирует признак верно
        df['count'] = df['libs'].map(len)

        # Удаляем неизвестные библиотеки (отсутствующие в train выборке, на которой делался fit)
        # Исходные данные и работа только с названиями ограничивают нас в возможностях
        # Была предпринята попытка сравнения названий неизвестных библиотек с известными при помощи
        # расстояния Левенштейна, но на валидации Recall по 1 остался прежним, а по 0 метрики понизились
        df.drop(columns=list(set(df['libs'].sum()) - set(self.libs_features)), inplace=True)

        return df


if __name__ == "__main__":
    # Загружаем данные
    df_train = pd.read_csv('train.tsv', sep='\t')

    # Предобработка данных
    preproc = Preprocess()
    preproc.fit(df_train)

    df_train = preproc.transform(df_train)
    df_train.pop('filename')
    df_train.pop('libs')

    # В качестве модели возьмем классификатор CatBoost
    # В течении 1000 эпох будем обучать ансамбль деревьев глубиной 10
    # Регуляризация на листе улучшает точность прогноза (повышается Recall по 1 и f1)
    model = CatBoostClassifier(iterations=1000,
                               depth=10,
                               l2_leaf_reg=0.5)

    # Обучаем модель
    target = df_train.pop('is_virus')
    model.fit(X=df_train, y=target)

    # Записываем модель в файл
    filename = 'malware_detector'

    model.save_model(filename)

    print(f"Model trained! File '{filename}' created!")
