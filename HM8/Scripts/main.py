import json
import pandas as pd

#Работаем с признаками ищем подход который сократит количество признаков и как с этим работать.
from sklearn.feature_extraction.text import CountVectorizer #попробовал лучший результат 77.8% на СatBoost
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

#общее
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#модели которые планируем обучить
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import (
    GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
import catboost as cb

from sklearn.linear_model import LogisticRegression #для меты

#закоментировал потому что перешли на лейбл энкодил csv
# with open('HM8/Resources/train.json', 'r') as f:
#     data = json.load(f)

# df = pd.DataFrame(data)


#-------------------------
#print("Vectorizer start")
# df['ingredients_str'] = df['ingredients'].apply(lambda x: ' '.join(x))
# vectorizer = CountVectorizer(min_df=10) 
# X = vectorizer.fit_transform(df['ingredients_str'])
#print("Vectorizer end")
#-------------------------

print("Read encoded_dataset start")
df_encoded = pd.read_csv('HM8/Resources/encoded_dataset.csv')
X = df_encoded.drop('cuisine', axis=1)
y = df_encoded['cuisine']
print("Read encoded_dataset end")

print("SelectKBest start")
k = 2500  
selector = SelectKBest(score_func=f_classif, k=k)
X_selected = selector.fit_transform(X, y)
selected_features_mask = selector.get_support()
selected_features = X.columns[selected_features_mask]
print("SelectKBest end")

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)


estimators = [
    #('gb', GradientBoostingClassifier(random_state=42)),
    #('ada', AdaBoostClassifier(random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=100, random_state=42)),
    ('knn', KNeighborsClassifier()),
    #('dt', DecisionTreeClassifier(random_state=42)),
    ('cat', cb.CatBoostClassifier(verbose=0, random_state=42)),
    #('svm', SVC(kernel='linear', random_state=42)),
    #('dummy', DummyClassifier(strategy="most_frequent"))
]

meta_model = LogisticRegression()

stacking_clf = StackingClassifier(estimators=estimators, final_estimator=meta_model)

stacking_clf.fit(X_train, y_train)

y_pred = stacking_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Точность стекинг-классификатора: {accuracy:.4f}")
print("Отчет о классификации:")
print(report)



# models = {
#     #"Gradient Boosting": GradientBoostingClassifier(random_state=42),
#     #"Ada Boost": AdaBoostClassifier(random_state=42),
#     #"Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
#     #"K Neighbors": KNeighborsClassifier(),
#     #"Decision Tree": DecisionTreeClassifier(random_state=42),
#     #"CatBoost": cb.CatBoostClassifier(verbose=0, random_state=42),
#     #"Dummy Classifier": DummyClassifier(strategy="most_frequent"),
#     "SVM (Linear Kernel)": SVC(kernel='linear', random_state=42)
# }

# results = []

# for name, model in models.items():
#     print(f"Обучение модели: {name}")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred, output_dict=True)
#     results.append({
#         "Model": name,
#         "Accuracy": accuracy,
#         "Precision": report['weighted avg']['precision'],
#         "Recall": report['weighted avg']['recall'],
#         "F1-Score": report['weighted avg']['f1-score']
#     })

# results_df = pd.DataFrame(results)

# print(results_df.sort_values(by="Accuracy", ascending=False))