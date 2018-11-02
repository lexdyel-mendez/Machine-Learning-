import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

train_df = pd.read_csv(r"C:\Users\lexdy\Desktop\MachineLearningGoogle\DataSets\Titanic\train.csv")
test_df = pd.read_csv(r"C:\Users\lexdy\Desktop\MachineLearningGoogle\DataSets\Titanic\test.csv")

print("Show the head of the train data \n")
print(train_df.head(),'\n')

print("Number of samples into train data is {}.".format(train_df.shape[0]),'\n')

print("Show the dead of the test data\n")
print(test_df.head(), '\n')

print("Number of samples into test data is {}".format(test_df.shape[0]), '\n')

print("Check for missing values \n")
print(train_df.isnull().sum(), '\n')

print("Percentage of missing Age records is %.2f%%" %((train_df["Age"].isnull().sum()/train_df.shape[0])*100) , '\n')

ax = train_df['Age'].hist(bins=15, density = True, stacked = True, color = 'teal', alpha = 0.6)
train_df['Age'].plot(kind = 'density', color = 'teal')
ax.set(xlabel = 'Age')
plt.xlim(-10, 85)
#plt.show()

print('The mean age is %.2f'%(train_df['Age'].mean(skipna=True)),'\n')
print('The median age is %.2f'%(train_df['Age'].median(skipna=True)),'\n')

print('Percentage of missing cabin records %.2f'%(train_df['Cabin'].isnull().sum()/train_df.shape[0]*100), '\n')

print('Percentage of missing embarked records %.2f'%(train_df['Embarked'].isnull().sum()/train_df.shape[0]*100), '\n')

print('Boarded passangers grouped by port of embarkation (C = Cherbourg, Q=Queenstown, S=Southampton):')
print(train_df['Embarked'].value_counts())
sns.countplot(x='Embarked',data = train_df, palette='Set2')
#plt.show()

print('The most common boarding port of embarkation is %s.' %train_df['Embarked'].value_counts().idxmax() , '\n')

train_data = train_df.copy()
train_data['Age'].fillna(train_df['Age'].median(skipna=True), inplace=True)
train_data["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)

print('Recheack for missing data')
print(train_data.isnull().sum() , '\n')

print('head of new data set: \n')
print(train_data.head())

plt.figure(figsize=(15,8))
ax = train_df['Age'].hist(bins = 15, density = True, stacked=True, color='teal',alpha = 0.6)
train_df['Age'].plot(kind='density',color='teal')
ax = train_data['Age'].hist(bins=15, density=True, stacked=True, color = 'orange', alpha =0.5)
train_data['Age'].plot(kind='density',color='orange')
ax.legend(['Raw Age', 'Adjusted Age'])
ax.set(xlabel = 'Age')
plt.xlim(-10,85)
#plt.show()

train_data['TravelAlone'] = np.where((train_data['SibSp']+train_data['Parch'])>0,0,1)
train_data.drop('SibSp',axis=1, inplace=True)
train_data.drop('Parch',axis=1, inplace=True)

training = pd.get_dummies(train_data, columns=['Pclass', 'Embarked', 'Sex'])
training.drop('Sex_female' , axis =1, inplace=True)
training.drop('PassengerId', axis =1, inplace = True)
training.drop('Name', axis=1, inplace = True)
training.drop('Ticket',axis=1, inplace = True)

final_train = training
print('The final training set: \n')
print(final_train.head(), '\n')

print('All the missing data from the test set \n')
print(test_df.isnull().sum(),'\n')

test_data = test_df.copy()
test_data['Age'].fillna(train_df['Age'].median(skipna=True), inplace = True)
test_data['Fare'].fillna(train_df['Fare'].median(skipna=True),inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

test_data['TravelAlone'] = np.where((test_df['SibSp']+test_df['Parch'])> 0, 0, 1)

test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)

testing = pd.get_dummies(test_data, columns=['Pclass', 'Embarked','Sex'])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
print('This is the final test set:\n')
print(final_test.head(),'\n')
print('New missing files for test (none) :\n', final_test.isnull().sum() , '\n')
