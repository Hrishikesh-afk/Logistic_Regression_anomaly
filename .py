%pip install seaborn
import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv('anomaly.csv')
sns.heatmap(df.isnull())
df.Col1=['H' if value >=0.5 else 'L' for value in df.Col1]
df['Col1']=df['Col1'].astype('category')
df['Col1'].value_counts().plot(kind='bar')
X=df.drop(['Col1'],axis=1)
y=df['Col1']
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test=train_test_split(X_scaled,y,test_size=0.30,random_state=42)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy:{accuracy:.2f}')
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
