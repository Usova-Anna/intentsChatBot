import random
import urllib.request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier




#url = "https://drive.google.com/uc?export=view&id=1u4sNekGHaDzgkOVzCOAbyWpFTEMfu95Z"
#url="https://drive.google.com/file/d/17Eqk5XBZoUtgNTk0CIFhHe-JFBwodSNX/view?usp=sharing"
url="https://drive.google.com/uc?export=view&id=17Eqk5XBZoUtgNTk0CIFhHe-JFBwodSNX"
filename = "intents_dataset.json"
urllib.request.urlretrieve(url, filename)

import json
with open(filename,'r',encoding='UTF-8') as file:
  data=json.load(file)


X=[]
y=[]
for name in data:
  for phrase in data[name]['examples']:
    X.append(phrase)
    y.append(name)
  for phrase in data[name]['responses']:
    X.append(phrase)
    y.append(name)
print(y)

vectorizer = CountVectorizer()
vectorizer.fit(["мама мыла раму","Саша мыла раму",]) #обучаем векторайзер
print(vectorizer.get_feature_names())
# print(vectorizer.transform(["мыла мыла раму"]).toarray())
# print(vectorizer.transform(["шла Саша по шоссе"]).toarray())


vectorizer=CountVectorizer()
vectorizer.fit(X)
print(vectorizer.get_feature_names())
#преобразуем СПИСОК фраз в набор чисел, представляющих встречаемость каждого слова
X_vec = vectorizer.transform(X)
print(X_vec.toarray())

print(vectorizer.transform(['какая погода']).toarray())

#создаём модель
model_mlp = MLPClassifier()

#обучаем модель
model_mlp.fit(X_vec,y)

#доля правильных ответов
accuracy = model_mlp.score(X_vec,y)
print(accuracy)

#получаем интент с помощью ML
MODEL= model_mlp
def get_intent(text):
  #сначала преобразуем в числа
  text_vec = vectorizer.transform([text])
  return model_mlp.predict(text_vec)[0] #Берём нулевой элемент, чтобы избавиться от формата "список"

def get_response(intent):
  return random.choice(data[intent]['responses'])

def bot(text):
  intent=get_intent(text)
  answer = get_response(intent)
  return answer

text=""
while text != "Выход":
  text = input("<")
  print(bot(text))






