import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from train import Preprocess


if __name__ == "__main__":
    # Загружаем модель из файла
    model = CatBoostClassifier()
    model.load_model('malware_detector')

    # Загружаем данные
    df_val = pd.read_csv('val.tsv', sep='\t')
    df_train = pd.read_csv('train.tsv', sep='\t')

    # Предобработка данных
    preproc = Preprocess()
    preproc.fit(df_train)

    df_val = preproc.transform(df_val)
    df_val.pop('libs')
    df_val.pop('filename')
    y_true = df_val.pop('is_virus')

    # Делаем прогноз
    y_pred = model.predict(df_val)

    # Используя sklearn, рассчитываем метрики
    tn, fp, fn, tp = confusion_matrix(y_true=y_true,
                                      y_pred=y_pred).ravel()

    report = classification_report(y_true=y_true,
                                   y_pred=y_pred,
                                   output_dict=True)

    # Записываем результаты в файл
    filename = 'validation.txt'

    with open(filename, 'w') as f:
        f.write(f"True positive: {tp}\n"
                f"False positive: {fp}\n"
                f"False negative: {fn}\n"
                f"True negative: {tn}\n"
                f"Accuracy: {round(report['accuracy'], 2)}\n"
                f"Precision: {round(report['1']['precision'], 2)}\n"
                f"Recall: {round(report['1']['recall'], 2)}\n"
                f"F1: {round(report['1']['f1-score'], 2)}\n")

    print(f"Validation done! File {filename} created!")
