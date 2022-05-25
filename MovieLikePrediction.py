#Çok Değişkenli Doğrusal Regresyon (Multiple Linear Regression) Projesi

import pandas #Pandas kütüphanesi ile veri işleme gerçekleştiricez. 
from sklearn import linear_model #Sklearn modülünü regresyon işlemi için kullanacağız.

df = pandas.read_csv("movie_data.csv") #Pandas kütüphanesi ile data setini import ediyoruz.

x = df[['Views', 'CommentCount']] #İki değişkene dayalı bir değeri tahmini etme modeli için bağımsız iki değişken seçildi.
y = df['Likes'] #Tahmin edeceğimiz bağımlı değişken seçildi.

regr = linear_model.LinearRegression() #LinearRegression() yöntemini doğrusal bir regresyon nesnesi oluşturmak için kullanıyoruz.
regr.fit(x, y) #fit() bağımsız ve bağımlı değerleri parametre olarak alan ve regresyon nesnesini tanımlanan verilere atayan yöntemdir.

likes = regr.predict([[2819118, 20573]]) #Filmin görüntelenme ve yorum sayısına dayalı olarak tahmin etmeye hazır bir regresyon modelimiz var.

print('Tahmini Beğenilme : ' , format(int(likes),',d')) #Beğeni tahmini ekranda gösteriliyor.

