import random
import urllib.request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
# Update-информация с сервера(новые сообщения, новые контакты)
from telegram import Update
from telegram.ext import ApplicationBuilder  # для создания и настройки ТГ-бота
from telegram.ext import MessageHandler  # обработчик создаёт реакцию(функцию) на действие
from telegram.ext import filters  # чтобы обрабатывать типы запросов, на которые будет срабатывать MessageHandler
import json

url = "https://drive.google.com/uc?export=view&id=17Eqk5XBZoUtgNTk0CIFhHe-JFBwodSNX"
filename = "intents_dataset.json"
urllib.request.urlretrieve(url, filename)

with open(filename, 'r', encoding='UTF-8') as file:
    data = json.load(file)

X = []
y = []
for name in data:
    for phrase in data[name]['examples']:
        X.append(phrase)
        y.append(name)
    for phrase in data[name]['responses']:
        X.append(phrase)
        y.append(name)
print(y)

vectorizer = CountVectorizer()
vectorizer.fit(["мама мыла раму", "Саша мыла раму", ])  # обучаем векторайзер
print(vectorizer.get_feature_names_out())
# print(vectorizer.transform(["мыла мыла раму"]).toarray())
# print(vectorizer.transform(["шла Саша по шоссе"]).toarray())


vectorizer = CountVectorizer()
vectorizer.fit(X)
print(vectorizer.get_feature_names_out())
# преобразуем СПИСОК фраз в набор чисел, представляющих встречаемость каждого слова
X_vec = vectorizer.transform(X)
print(X_vec.toarray())

print(vectorizer.transform(['Как дела']).toarray())
print("dfsdfsdf")
# создаём модель
model_mlp = MLPClassifier()

# обучаем модель
model_mlp.fit(X_vec, y)

# доля правильных ответов
accuracy = model_mlp.score(X_vec, y)
print(accuracy)

# получаем интент с помощью ML
MODEL = model_mlp


def get_intent(text):
    # сначала преобразуем в числа
    text_vec = vectorizer.transform([text])
    return model_mlp.predict(text_vec)[0]  # Берём нулевой элемент, чтобы избавиться от формата "список"


def get_response(intent):
    return random.choice(data[intent]['responses'])


def bot(text):
    intent = get_intent(text)
    answer = get_response(intent)
    return answer


TOKEN = '5588469713:AAH36_LzeBtlIE5fbzxM6-JVzpvRdvOx6Ac'


# функция для MessageHandler'а, вызывать её при каждом сообщении боту. :Тип_параметра
async def reply(update: Update, context) -> None:
    user_text = update.message.text
    reply = bot(user_text)
    print("<", user_text)
    print(">", reply)
    await update.message.reply_text(reply)  # вместо return, ответ пользователю в чат

phrase = ""
while phrase != "Exit":
    phrase = input("<")
    print(bot(phrase))



# создаём объект приложения - связываем функцию с ботом
app = ApplicationBuilder().token(TOKEN).build()


# создаём обработчик текстовых сообщений
handler = MessageHandler(filters.Text(), reply)

# добавляем обработчик в приложение
app.add_handler(handler)

# запускаем приложение
app.run_polling()

