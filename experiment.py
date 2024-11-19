import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from smote_extension import main as smote_hybrid
from sklearn.model_selection import train_test_split


df = pd.read_csv('camel.csv')
print(df)

df_sampled = smote_hybrid(df, 200, 10)
print(df_sampled)

x = df_sampled.drop(columns=['bug'])
y = df_sampled['bug']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print("Training shape:", x_train.shape," || Testing shape:", y_train.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
print(y_pred)

acc = accuracy_score(y_pred, y_test)
print('accuracy:',acc)

prec = precision_score(y_pred, y_test)
print('precision:',prec)

f1 = f1_score(y_pred, y_test)
print('f1:',f1)