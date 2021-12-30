import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

feature_column = 'desired2'
raw_dataset = pd.read_csv('Credit_Screening.dat',
                          header=0, encoding='utf-8', delimiter=';', decimal='.', escapechar=' ')
raw_data = raw_dataset.drop(feature_column, axis=1)

scaler = StandardScaler()
scaler.fit(raw_data)
scaled_features = scaler.transform(raw_data)
scaled_data = pd.DataFrame(scaled_features, columns=raw_data.columns)

x = scaled_data
y = raw_dataset[feature_column]
y = y.astype('int')

x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size=0.2)

error_rates = []

for i in np.arange(1, 101):
    new_model = KNeighborsClassifier(n_neighbors=i)
    new_model.fit(x_training_data.values, y_training_data)
    new_predictions = new_model.predict(x_test_data.values)
    error_rates.append(np.mean(new_predictions != y_test_data))

plt.plot(error_rates)
plt.xlabel("значение k")
plt.ylabel("доля неверных предсказаний")
plt.tight_layout()
plt.show()

model = KNeighborsClassifier(n_neighbors=30)
model.fit(x_training_data.values, y_training_data)
predictions = model.predict(x_test_data.values)
cm = confusion_matrix(y_test_data, predictions)
print(classification_report(y_test_data, predictions))
print(cm)
print('error=', np.mean(predictions != y_test_data))

ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
label_arr = ['Accepted Credits', 'Declined Credits']
ax.xaxis.set_ticklabels(label_arr)
ax.yaxis.set_ticklabels(label_arr)
plt.show()
