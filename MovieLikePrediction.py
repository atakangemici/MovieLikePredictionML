import pandas
from sklearn import linear_model

df = pandas.read_csv("movie_data.csv")

x = df[['Views', 'CommentCount']]
y = df['Likes']

regr = linear_model.LinearRegression()
regr.fit(x, y)

likes = regr.predict([[2564903, 22149]])

print('Tahmini Beğenilme : ' , format(int(likes),',d'))

