#import the nned lib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the data
titanic_data = pd.read_csv('tested.csv')

# preprocessing
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] #value for ref
target = 'Survived'  #data value that we targeting
X = titanic_data[features]
y = titanic_data[target]


# Converting variables to numerical using one-hot encoding
X = pd.get_dummies(X)

# missing values fill with mean
X = X.fillna(X.mean())

# Train the data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict survival status of all passengers
predictions = model.predict(X)
#print(predictions)

# Add the predictions to the dataset
titanic_data['Prediction'] = predictions


result_data = titanic_data[['PassengerId', 'Survived', 'Prediction']]

result_data.to_csv('resultoftitanicdata.csv',index=False)
print("done")
