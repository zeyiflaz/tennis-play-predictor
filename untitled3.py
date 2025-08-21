# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 14:06:59 2025

@author: user
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")
print(veriler)
outlook = veriler.iloc[:,0:1].values #satırın tümünü aldı sütunda 0. indexi aldı
print(outlook)

from sklearn import preprocessing # burada veriler içeri aktarılıyor bu kütüphane katagorik verileri sayısal verilere çevirmeye yarar

le = preprocessing.LabelEncoder() # buraya veriler sayıya dönüştürülür

outlook[:,0] = le.fit_transform(veriler.iloc[:,0]) # ülke sutunu aliniyor daha sonra fit ile model egitiliyor ve ulke adlari sayiya donusuyor
print(outlook)
# sadece LabelEncoder ile dönüşüm yapılırsa katagorileri sayiya cevirir fakat bu cevirimler sira var gibi gozukur(ör: tr=0, us=1, fr=2) buda sikintilara yol acar o yuzden OneHotEncoder kullanılır
#OneHotEncoder: her katagoriye ayri sutun verir, 0-1 degerle kodlar.

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray() # outlooktakileri kolona 3 cevirir
print(outlook)

p= veriler.iloc[:,-1:].values 
print(p)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

p[:,-1]= le.fit_transform(veriler.iloc[:,-1])
print(p)

ohe = preprocessing.OneHotEncoder()
p = ohe.fit_transform(p).toarray()
print(p)

# daha onceden one-hot encoder ile donusturulen NumPy dizisi pandas tablosune(dataFrame) ceviriliyor.
sonuc = pd.DataFrame(data= outlook, index = range(14), columns=['sunny','overcast','rainy']) 
print(sonuc)

humidity = veriler.iloc[:,1:4].values
sonuc2 = pd.DataFrame(data= humidity, index= range(14), columns= ['emperaty','humidity','windy'])
print(sonuc2)

play = veriler.iloc[:,-1].values
print(play)

sonuc3 = pd.DataFrame(data= p[:,:1], index= range(14), columns= ['play'])
print(sonuc3)


# dataframe birleştirme işlemi 
# axis= 0 satir basli birlestir
#axis=1 yatay sütün olarak birlestir anlamina gelir

s = pd.concat([sonuc,sonuc2], axis=1)
print(s)
s2 = pd.concat([s,sonuc3], axis=1)
print(s2)

# verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3, test_size=0.33, random_state=0)

# verilerin olceklendirilmesi

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #lineer regresyon modeli olusturuluypr
regressor.fit(x_train,y_train) #model egitim verisi ile ogreniyor

#tahmin etme
y_pred= regressor.predict(x_test) # x_test verileri modele veriliyor model bu verilere karsilik ne tahmin ediyorsa onları yaziyor.

emparaty = s2.iloc[:,3:4].values
print(emparaty)
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]
veri = pd.concat([sol,sag],axis=1)
x_train, x_test, y_train, y_test = train_test_split(veri,emparaty,test_size = 0.33, random_state=0)
r2 = LinearRegression()
r2.fit(x_train,y_train)
y_pred = r2.predict(x_test)

from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

le = LabelEncoder()
for column in veri.columns:
    if veri[column].dtype == 'object':
        veri[column] = le.fit_transform(veri[column])

# X (bagimsiz degiskenler)
X_1 = sm.add_constant(veri.values.astype(float))  # sabit terim eklendi

# Y (bagimsiz degiskenler) – temperature (emparaty)
Y_1 = s2.iloc[:,3].values.astype(float)

# OLS modeli
model = sm.OLS(Y_1, X_1).fit()
print(model.summary())


  


 