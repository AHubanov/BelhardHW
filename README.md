# BelhardHW

#HM8
Код находится в одном из промежуточных состоянии. 
Что я попробовал в этом ДЗ:
    Работал с признаками. Использовал vectorizer. Я сомневался в данном подходе, потому что вектор строится по очень замусоренным ингридиентам + перед подсчетом вектора все склеивалось воедино. Результат работы моделей на базе векторизации признаков на скрине 0_result_vectorizer.png.

    Попробовал почистить признаки от спецсимволов, комментариев и пробелов. И применил LabelEncoder. LabelEncoder работает долго, поэтому подготовил промежуточный результат в виде encoded_dataset.csv. Кстати даже его загрузка оказалась долгой.

    Результат работы моделей после LabelEncoder на скринах 1_result_labelencoding_1000K_of_signs и 1_result_labelencoding_2000K_of_signs. (Так как обучение моделей занимало много времени, то их список заметно сократился). Промежуточные тесты стал проводить на 1-3 моделях. 

    Список моделей на которых запустил работу dataset хотя бы один раз:
        GradientBoostingClassifier(random_state=42),
        AdaBoostClassifier(random_state=42),
        ExtraTreesClassifier(n_estimators=100, random_state=42),
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=42),
        cb.CatBoostClassifier(verbose=0, random_state=42),
        DummyClassifier(strategy="most_frequent"),
        SVC(kernel='linear', random_state=42)

    Запустил стекинг в ансамбле 3 модели:
            ExtraTreesClassifier(n_estimators=100, random_state=42)
            KNeighborsClassifier()
            cb.CatBoostClassifier(verbose=0, random_state=42)
    + мета модель представлена LogisticRegression
    
    Результат в 3_result_stacking.png 

    Лучший результат получен vectorizer признаков на CatBoost.
        
Что не успел попробовать:
    Так как модельки обучались очень долго, то хотел попробовать подход с сохранением модели и поднятие из сейва. 
    Хотел попробовать использовать Grid для подбора оптимальных параметров. Но моделей слишком много и данное действие не представлялось возможным. 
    Привести исходные данные в прекрасный порядок. С ингридиентами для блюд можно придумать много интересного, чтобы оптимизировать.

    
