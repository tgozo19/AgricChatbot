import json
import pickle
import random

import nltk
import numpy
import tensorflow
import tflearn
from flask import Flask, request
from nltk.stem.lancaster import LancasterStemmer
from twilio.rest import Client

app = Flask(__name__)

account_sid = 'ACac45698f788c44f4d3af93c33d2ada80'
auth_token = 'dead486dd0b5e1b25e9bf5516fba09fe'
client = Client(account_sid, auth_token)


@app.route('/')
def home():
    message = client.messages.create(
        from_='whatsapp:+14155238886',
        body='Your appointment is coming up on July 21 at 3PM',
        to=f'whatsapp:+{phone_number}'
    )
    return 'Hello, World! ' + message.sid


@app.route('/sms', methods=['GET', 'POST'])
def hello_world():  # put application's code here
    user_message = request.form.get("Body")
    phone_number = request.form.get("WaId")
    nltk.download('punkt')
    stemmer = LancasterStemmer()

    with open("intents.json") as file:
        data = json.load(file)

    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    except:
        words = []
        labels = []
        docs_x = []
        docs_y = []

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))

        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [stemmer.stem(w.lower()) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

        training = numpy.array(training)
        output = numpy.array(output)

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

    tensorflow.compat.v1.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    percentage = 0

    try:
        if user_message == 'fucken train this model':
            abc = client.messages.create(
                from_='whatsapp:+14155238886',
                body='Training the model',
                to=f'whatsapp:+{phone_number}'
            )
            model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
            score = model.evaluate(training, output)
            percentage = score[0]
            while percentage < 0.8:
                model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
                score = model.evaluate(training, output)
                percentage = score[0]
            model.save("model.tflearn")

            efg = client.messages.create(
                from_='whatsapp:+14155238886',
                body=f'Finished training the model an score is {percentage}',
                to=f'whatsapp:+{phone_number}'
            )

            return '0'
        model.load("model.tflearn")
    except:
        abc = client.messages.create(
            from_='whatsapp:+14155238886',
            body='Training the model',
            to=f'whatsapp:+{phone_number}'
        )
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        score = model.evaluate(training, output)
        percentage = score[0]
        while percentage < 0.8:
            model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
            score = model.evaluate(training, output)
            percentage = score[0]
        model.save("model.tflearn")

        efg = client.messages.create(
            from_='whatsapp:+14155238886',
            body=f'Finished training the model an score is {percentage}',
            to=f'whatsapp:+{phone_number}'
        )

        return '0'

    def bag_of_words(s, words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return numpy.array(bag)

    def chat(inp):

        results = model.predict([bag_of_words(inp, words)])

        results_index = numpy.argmax(results)

        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                return random.choice(responses)

        return "Didn't understand"

    msg = chat(user_message)

    message = client.messages.create(
        from_='whatsapp:+14155238886',
        body=msg,
        to=f'whatsapp:+{phone_number}'
    )

    return '1'


if __name__ == '__main__':
    app.run(debug=True)
