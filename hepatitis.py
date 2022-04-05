import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


st.title("Machine Learning - Classification")
st.sidebar.write("""This is a demo app using libraries such as Streamlit, sklearn etc""")
st.sidebar.write("This [data set](https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset) contains laboratory values of blood donors and Hepatitis C patients")
st.sidebar.write("For more information please contact: ")
st.sidebar.write("[Mohd Noor Syawal Faqqie](https://www.linkedin.com/in/mohd-noor-syawal-faqqie-abdullah-b8764b236/)")


df =pd.read_csv(r'https://raw.githubusercontent.com/SyawalFaqqie/sklearn-vizz/main/HepatitisCdata.csv')


imputer = SimpleImputer(missing_values=np.nan,strategy="median")
imputer.fit(df.iloc[:,[4,5,6,10,13]])
df.iloc[:,[4,5,6,10,13]]=imputer.transform(df.iloc[:,[4,5,6,10,13]])

labelencoder1 = LabelEncoder()
labelencoder2 = LabelEncoder()
df['Category'] = labelencoder1.fit_transform(df['Category'])
df['Sex'] = labelencoder1.fit_transform(df['Sex'])

X=df.iloc[:,2:].values
y=df.iloc[:,1].values


classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('SVM', 'KNN','Random Forest Classifier','Logistic Regression')
  
)
test_data_ratio = st.sidebar.slider('Select testing size or ratio',
                                    min_value = 0.10,
                                    max_value = 0.50,
                                    value=0.2
)

random_state = st.sidebar.slider('Select random state', 1,9999,value=1234)

st.write("## 1: Summary (X variables)")
st.write('Shape of predictors @ X variables :', X.shape)
st.write('Summary of predictors @ X variables:', pd.DataFrame(X).describe())

st.write("## 2: Summary of y varible")
yclass=len(np.unique(y))
st.write('Number of classes: ',yclass)



def add_parameter_ui(clf_name):
  params=dict()
  if clf_name=='SVM':
    C=st.sidebar.slider('C',0.01,10.0,value=1.0)
    params['C']=C
  elif clf_name=='KNN':
    K=st.sidebar.slider('K',1,15,value=5)
    params['K']=K
  elif clf_name == 'Random Forest Classifier':
    n_estimators=st.sidebar.slider('n_estimators',1,100,value=10)
    params['n_estimators']=n_estimators
    max_depth=st.sidebar.slider('max_depth',2,15,value=5)
    params['max_depth']=max_depth
  else:
    
  return params
params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'],kernel="rbf")
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'],metric="minkowski",p=2)
    elif clf_name == 'Random Forest Classifier':
        
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'],criterion="entropy", random_state=random_state)
    else:
        clf = LogisticRegression(random_state=random_state)
    return clf

clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data_ratio, random_state=random_state)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)    

clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
st.write('## 2:Classifier: ',classifier_name)
st.write('Classification Report')

report=classification_report(y_test,y_pred,output_dict=True)
data=pd.DataFrame(report).transpose()
st.write(data)

