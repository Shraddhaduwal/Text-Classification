import os
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def read_files(path):
    documents, labels = [], []
    for directory in os.listdir(path):
        for file in os.listdir(path + '/' + directory):
            try:
                text = ""
                filename = path + '/' + directory + '/' + file
                with open(filename, 'r', encoding='utf-8') as f:
                    text += f.read()
                documents.append(text)
                labels.append(directory)
            except Exception:
                pass
    return documents, labels


stopWords = set(nltk.corpus.stopwords.words('english'))

print("reading files")
docs, labs = read_files("/home/shraddha/Desktop/bbcnews/data_files")
x_train, x_test, y_train, y_test = train_test_split(docs, labs, test_size=0.25)


tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(" "),
                                   sublinear_tf=True, encoding="utf-8",
                                   decode_error='ignore',
                                   max_df=0.5,
                                   min_df=10,
                                   stop_words=stopWords)

vectorized = tfidf_vectorizer.fit_transform(x_train)
print('No of Samples , No. of Features ', vectorized.shape)

# Classifier

clf1 = Pipeline([('vect', tfidf_vectorizer),
                 ('clf', MultinomialNB())])

clf2 = Pipeline([('vect', tfidf_vectorizer),
                 ('clf', SVC(kernel='linear'))])


def train_evaluate(clf, xtr, xte, ytr, yte):
    print(len(xtr), len(ytr))
    clf.fit(xtr, ytr)
    print("Accuracy of train data set", clf.score(xtr, ytr))
    print("Accuracy of test data set", clf.score(xte, yte))
    pred = clf.predict(xte)
    print("Accuracy of predicted test data set", pred)

    some_data = " I won gold medal in basketball."
    red = clf.predict([some_data])
    print("Some prediction", red)


print("Multinomial Naive Bayes:")
train_evaluate(clf1, x_train, x_test, y_train, y_test)

print("SVM")
train_evaluate(clf2, x_train, x_test, y_train, y_test)
