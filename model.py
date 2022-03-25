# classification prediction model
import pandas as pd
import numpy as np
import pickle
data_test=pd.read_csv('MobileTest.csv')
data_train=pd.read_csv('MobileTrain.csv')
#outlier detection
data = pd.concat([data_train.assign(ind="train"), data_test.assign(ind="test")])
ax=data.select_dtypes(include=['int','float64'])
data=data.drop('id',axis=1)
ax=['fc','px_height']
def outlier_det(col_name,data):
    Q1=np.percentile(data[col_name],25,interpolation='midpoint')
    Q2=np.percentile(data[col_name],50,interpolation='midpoint')
    Q3=np.percentile(data[col_name],75,interpolation='midpoint')
    IQR=Q3-Q1
    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR
    new_df = data[(data[col_name] < upper) & (data[col_name] > lower)]
    return new_df
#splitting as train and test
test,train=data[data['ind'].eq("test")],data[data['ind'].eq("train")]
test=test.drop('ind',axis=1)
train=train.drop('ind',axis=1)
#splitting x and y
Y=train['price_range']
X=train.drop(['price_range','touch_screen','four_g','wifi','dual_sim','three_g','fc','n_cores','blue','pc'],axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=42,test_size=.2)
#modeling
from sklearn.svm import SVC
svm_linear=SVC(kernel='linear')
a25=svm_linear.fit(x_train,y_train)

# Saving the model
pickle.dump(a25,open('mobile.pkl','wb') )


