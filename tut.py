import machine_learning_library1 as mll1
import pandas as pd
import numpy as np
from machine_learning_library1 import DecisionTree
from machine_learning_library1 import knn
##predictions of linear_test
df= pd.read_csv(r"C:\Users\Krishna\Desktop\modules\python_tutorial\linear_regression_test.csv")
df1 = pd.read_csv(r"C:\Users\Krishna\Desktop\modules\python_tutorial\linear_regression_train.csv")
x = df.iloc[:,1:25]
x = np.array(x)
x1 = np.array(df1.iloc[:,1:25])
y= np.array(df1.iloc[:,26])
x1_train  = np.array(x1[0:28800])
x1_cv = np.array(x1[28801:38400])
x1_test = np.array(x1[38401:48001])
y_train  = np.array(x1[0:28800])
y_cv = np.array(x1[28801:38400])
y_test = np.array(x1[38401:48001])
mll1.mvlr(x1,y,50,0.0000001)
mll1.predict_linr(x1_test,x1_cv,x1_train , y_test,y_cv,y_train)
mll1.apply_test(x)
##predictions on polynomial test set
df = pd.read_csv(r"C:\Users\Krishna\Desktop\modules\python_tutorial\polynomial_regression_train.csv")
df1 = pd.read_csv(r"C:\Users\Krishna\Downloads\polynomial_regression_test.csv")
x = df["Feature_1"]
y = df['Target']
x1 = df1['Feature_1'] #pick any feature and check predicted target 
mll1.pr(x,y,3) #change degree to increase to r sqaured error also graph here can be seen when algo is run
mll1.pass_testset(x1)
##predictions on binary classfication
df = pd.read_csv(r"C:\Users\Krishna\Downloads\binary_classification_train.csv")
df1= pd.read_csv(r"C:\Users\Krishna\Downloads\binary_classification_test.csv")
x = df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
x1 = df1.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
y = df.iloc[:,21]
model = DecisionTree(3)
model.fit(x,y)

pred = model.predict(x1)
print("pred:" , pred)


##predictions on multi classification
df = pd.read_csv(r"C:\Users\Krishna\Downloads\multi_classification_train.csv")
df1 = pd.read_csv(r"C:\Users\Krishna\Downloads\multi_classification_test.csv")
x = df.iloc[:,1:20]
x1 = df1.iloc[:,1:20]
y = df.iloc[:,21]
model = knn(5)
model.fit(x,y)
model.predict(x1)





































 









































    

   























