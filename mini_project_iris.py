import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

%matplotlib inline

df = pd.read_csv('E:\Iris.csv')

print("\n***********\n")
print(f"Sample Dataset :- \n {df.head()}")
print("****************")
print(f"Shape of the dataset :- {df.shape}")

print("\n***********\n")
print(f"Checking for whether null values present or not :- \n{df.isnull().sum()}")

print(f"Duplicate values :- {len(df.loc[df.duplicated()])}")


iris_data = df.drop_duplicates()
print("\n***********\n")
print(f"Target value counts :- \n {iris_data['Species'].value_counts()}")

print("\n****************")
print(f"\nSVM Classification Algorithm Implementation:-")

X = iris_data.drop(['Species'], axis=1)
y = iris_data['Species']

pipe_line = Pipeline([
                      ('std_scaler', StandardScaler())
])
 
X = pipe_line.fit_transform(X)

label_encode = LabelEncoder()

y = label_encode.fit_transform(y)

print(f"\nLabels list from label encoder :- {list(label_encode.classes_)}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(f"Size Of The Train Dataset :- {len(X_train)}")
print(f"Size Of The Test Dataset :- {len(X_test)}")

svc_clf = SVC(C=0.5, kernel='linear')
 
svm_model = svc_clf.fit(X_train, y_train)

predict_result = svc_clf.predict(X_test)

Confusion_matric = confusion_matrix(y_test, predict_result)
print(f"\nConfusion matrix :- \n {Confusion_matric}\n")

clf_report = classification_report(y_test, predict_result)
print(f"Classification Report :- \n {clf_report}")

#accuracy = accuracy_score(y_test, predict_result)
#print(f"Iris classificication Accuracy :- {accuracy}\n")


print(f"\n\nDecisionTree Classification Algorithm Implementation:-")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

encode = LabelEncoder()

x = df.drop('Species', axis = 1)
x.head()

y = df.Species
y

y_encoded = encode.fit_transform(y)
#print(y_encoded)

x_test, x_val, y_test, y_val = train_test_split(x,y_encoded,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_test, y_test)


y_pred = classifier.predict(x_val)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_val, y_pred)
cm

from sklearn.tree import DecisionTreeClassifier
gini_c = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
DT_Model = gini_c.fit(x_test, y_test)


y_pred = gini_c.predict(x_val)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_val, y_pred)
cm 

# Summary of the predictions made by the classifier
print(f"\nConfusion Matrix :- \n {confusion_matrix(y_val, y_pred)}\n")
print(f"\nClassification Report :- \n {classification_report(y_val, y_pred)}\n")
# Accuracy score
#print(f'\naccuracy is',accuracy_score(y_pred,y_val))



#KNN
print("\n\nKNN Classification Algorithm Implementation:-")
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = df[feature_columns].values
y = df['Species'].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Loading libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Instantiate learning model (k = 3)
classifier = KNeighborsClassifier(n_neighbors=3)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cm

# Summary of the predictions made by the classifier
print(f"\nConfusion Matrix :- \n {confusion_matrix(y_test, y_pred)}\n")
print(f"\nClassification Report :- \n {classification_report(y_test, y_pred)}\n")


print('\n*******Results******')

k_list = list(range(1,50,2))
# creating list of cv scores
cv_scores = []

# perform 10-fold cross validation
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

    
from sklearn.metrics import accuracy_score, log_loss
classifiers = [
    SVC(C=0.5, kernel='linear'),
    DecisionTreeClassifier(criterion = 'gini', random_state = 0),
    KNeighborsClassifier(n_neighbors=3),
 
                  ]
 
# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
 
for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    log_entry = pd.DataFrame([[name, acc*100, 11]], columns=log_cols)
    log = log.append(log_entry)
    
    print("="*30)