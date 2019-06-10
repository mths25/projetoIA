from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from joblib import load, dump
from sklearn.svm import LinearSVC

def _get_accuracy(matrix):
    acc = 0
    n = 0
    total = 0

    for i in range(0, len(matrix)):
        for j in range(0, len(matrix)):
            if(i == j):
                n += matrix[i,j]

            total += matrix[i,j]

    acc = n / total
    return acc

X_train , y_train, X_test, y_test = load('preprocessed.pkl')

model = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(smooth_idf=True, sublinear_tf=True, use_idf=True)),
    ('clf', LinearSVC(C=0.1)),
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
matrix = metrics.confusion_matrix(y_test, predictions)
acc = _get_accuracy(matrix)
print(acc)
print(metrics.classification_report(y_test, predictions, target_names=model.classes_))
dump(model, 'model.pkl')
