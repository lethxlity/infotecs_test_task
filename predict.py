import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from train import Preprocess


if __name__ == "__main__":
    # Загружаем модель из файла
    model = CatBoostClassifier()
    model.load_model('malware_detector')

    # Загружаем данные
    df_train = pd.read_csv('train.tsv', sep='\t')
    df_test = pd.read_csv('test.tsv', sep='\t')

    # Предобработка данных
    preproc = Preprocess()
    preproc.fit(df_train)

    df_test = preproc.transform(df_test)
    df_test_libs = df_test.pop('libs')

    # Делаем прогноз
    y_pred = model.predict(df_test)

    # Сохраняем результаты прогноза в файл
    filename = 'prediction.txt'

    # Так как на выходе классификатора - numpy.array,
    # то воспользуемся встроенным методом библиотеки numpy
    np.savetxt(fname=filename,
               X=y_pred,
               delimiter='\n',
               header='prediction',
               fmt='%d',
               comments='')

    print(f"Prediction made! File {filename} created!")

    # Рассчитываем важность признаков для трактовки результатов прогноза
    # Слайс [:-1] значит то, что мы берем все признаки из датафрейма,
    # кроме последнего (количество используемых библиотек)
    importance = pd.DataFrame(data=model.get_feature_importance(),
                              columns=['weight'])[:-1]
    importance['lib'] = preproc.libs_features

    # Берем только признаки с ненулевыми весами
    importance = importance[importance['weight'] > 0]

    # Записываем все в dict для последующего быстрого
    # доступа к значению веса по названию признака
    importance.set_index('lib', inplace=True)
    importance = importance.to_dict(orient='index')

    # Формируем файл с ответами для вывода
    ans = pd.DataFrame(y_pred, columns=['is_virus'])

    # Сохраняем результаты прогноза в файл
    filename = 'explain.txt'

    with open(filename, 'w', encoding='utf-8') as f:
        # Используется для форматирования вывода
        maxlen = len(str(max(ans.index)))
        # Итерируемся по каждому предсказанию
        for idx in ans.index:
            f.write('\n' + maxlen * 10 * '-')
            f.write('\n' + str(idx) + (maxlen - len(str(idx))) * ' ' + '|')

            if ans.iloc[idx]['is_virus'] == 1:
                # Вес (важность) каждого признака (библиотеки) в предсказании модели
                weights = {}
                # Сумма весов признаков на которых делается предсказание
                sum_weights = 0

                # Библиотека с самой большой важностью
                max_weighted_lib = ''
                # Вес этой библиотеки
                max_weight = 0

                for lib in df_test_libs.iloc[idx]:
                    # Если библиотека важна для предсказания (содержится в importance)
                    if lib in importance:
                        # Добавляем ее в словарь
                        weights[lib] = importance[lib]['weight']
                        sum_weights += importance[lib]['weight']
                        # Ищем библиотеку с самым большим весом
                        if weights[lib] > max_weight:
                            max_weight = weights[lib]
                            max_weighted_lib = lib
                # Проверка условия на то, что хоть одна библиотека имеет ненулевой вес
                if max_weight != 0:
                    # Сортируем по убыванию весов
                    weights = {lib: weight for lib, weight in sorted(weights.items(),
                                                                     key=lambda x: x[1],
                                                                     reverse=True)}
                    # Первый случай - единственная библиотека с большим весом
                    if max_weight == sum_weights:
                        f.write(' Единственная библиотека' + '\n' + maxlen * ' ' + '| ' + max_weighted_lib)
                    # Второй случай - библиотека с доминирующим весом относительно остальных
                    elif max_weight / sum_weights > 0.8:
                        f.write(' Подозрительная библиотека' + '\n' + maxlen * ' ' + '| ' + max_weighted_lib)
                    # Третий случай - примерно равновзвешенные относительно друг друга библиотеки
                    else:
                        f.write(' Подозрительная комбинация')
                        for lib, weight in weights.items():
                            f.write('\n' + maxlen * ' ' + '| ' + '-' + lib)

    print(f"Explanation given! File {filename} created!")
