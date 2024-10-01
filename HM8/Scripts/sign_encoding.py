import json
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

with open('HM8/Resources/train.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

def clean_ingredient(ingredient):
    ingredient = re.sub(r'[^a-zA-Z\s]', '', ingredient)
    ingredient = ingredient.strip()  
    return ingredient

df['ingredients'] = df['ingredients'].apply(lambda ingredients: [clean_ingredient(ingredient) for ingredient in ingredients])

df['ingredients'] = df['ingredients'].apply(lambda ingredients: [ingredient.lower() for ingredient in ingredients])

all_ingredients = sorted(set([ingredient for sublist in df['ingredients'] for ingredient in sublist]))

label_encoder = LabelEncoder()
label_encoder.fit(all_ingredients)

df['encoded_ingredients'] = df['ingredients'].apply(lambda ingredients: label_encoder.transform(ingredients))

mlb = MultiLabelBinarizer(classes=label_encoder.transform(all_ingredients))
X = mlb.fit_transform(df['encoded_ingredients'])

df_encoded = pd.DataFrame(X, columns=label_encoder.classes_)
df_encoded['cuisine'] = df['cuisine']

df_encoded.to_csv('HM8/Resources/encoded_dataset.csv', index=False)