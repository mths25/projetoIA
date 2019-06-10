import pandas as pd
import re
import nltk
from nltk.tokenize import wordpunct_tokenize
from joblib import dump

class Preprocess():
    def __init__(self):
        self.emoticon_list = {':))': 'positive_emoticon', ':)': 'positive_emoticon', ':D': 'positive_emoticon', ':(': 'negative_emoticon', ':((': 'negative_emoticon', '8)': 'neutral_emoticon'}
        self.std_list = {'eh': 'é', 'vc': 'você', 'vcs': 'vocês','tb': 'também', 'tbm': 'também', 'obg': 'obrigado', 'gnt': 'gente', 'q': 'que', 'n': 'não', 'cmg': 'comigo', 'p': 'para', 'ta': 'está', 'to': 'estou', 'vdd': 'verdade'}
        self.stopword_list = []
        nltk_stopwords = nltk.corpus.stopwords.words('portuguese')
        # get custom stopwords from a file (pt-br). You can create your own database of stopwords on a text file, mongodb, so on...
        df = pd.read_fwf('stopwords-pt.txt', header = None)
        # list of array
        custom_stopwords = df.values.tolist()
        # transform list of array to list
        custom_stopwords = [s[0] for s in custom_stopwords]

        # You can also add stopwords manually instead of loading from the database. Generally, we add stopwords that belong to this context.
        self.stopword_list.append('é')
        self.stopword_list.append('vou')
        self.stopword_list.append('que')
        self.stopword_list.append('tão')
        self.stopword_list.append('...')
        self.stopword_list.append('«')
        self.stopword_list.append('➔')
        self.stopword_list.append('|')
        self.stopword_list.append('»')
        self.stopword_list.append('uai') # 'expression from the mineiros (MG/Brazil)'

        # join all stopwords
        self.stopword_list.extend(nltk_stopwords)
        self.stopword_list.extend(custom_stopwords)
        # remove duplicate stopwords (unique list)
        self.stopword_list = list(set(self.stopword_list))

    def _apply_stemmer(self, tokens):
        ls = []
        stemmer = nltk.stem.RSLPStemmer()

        for tk_line in tokens:
            new_tokens = []

            for word in tk_line:
                word = str(stemmer.stem(word))
                new_tokens.append(word)

            ls.append(new_tokens)

        return ls

    def _untokenize_text(self, tokens):
        ls = []

        for tk_line in tokens:
            new_line = ''

            for word in tk_line:
                new_line += word + ' '

            ls.append(new_line)

        return ls

    def _remove_url(self, data):
        ls = []
        words = ''
        regexp1 = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        regexp2 = re.compile('www?.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

        for line in data:
            line = str(line)
            urls = regexp1.findall(line)

            for u in urls:
                line = line.replace(u, ' ')

            urls = regexp2.findall(line)

            for u in urls:
                line = line.replace(u, ' ')

            ls.append(line)
        return ls

    def _remove_regex(self, data, regex_pattern):
        ls = []
        words = ''

        for line in data:
            line = str(line)
            matches = re.finditer(regex_pattern, line)

            for m in matches:
                line = re.sub(m.group().strip(), '', line)

            ls.append(line)

        return ls

    def _replace_emoticons(self, data, emoticon_list):
        ls = []

        for line in data:
            line = str(line)
            for exp in emoticon_list:
                line = line.replace(exp, emoticon_list[exp])

            ls.append(line)

        return ls

    def _tokenize_text(self, data):
        ls = []

        for line in data:
            line = str(line)
            tokens = wordpunct_tokenize(line)
            ls.append(tokens)

        return ls

    def _apply_standardization(self, tokens, std_list):
        ls = []

        for tk_line in tokens:
            new_tokens = []

            for word in tk_line:
                if word.lower() in std_list:
                    word = std_list[word.lower()]

                new_tokens.append(word)

            ls.append(new_tokens)

        return ls

    def _remove_stopwords(self, tokens, stopword_list):
        ls = []

        for tk_line in tokens:
            new_tokens = []

            for word in tk_line:
                if word.lower() not in stopword_list:
                    new_tokens.append(word)

            ls.append(new_tokens)

        return ls

    def run(self, X):
        X = self._remove_url(X)
        regex_pattern = '#[\w]*'
        X = self._remove_regex(X, regex_pattern)
        regex_pattern = '@[\w]*'
        X = self._remove_regex(X, regex_pattern)
        X = self._replace_emoticons(X, self.emoticon_list)
        X = self._tokenize_text(X)
        X = self._apply_standardization(X, self.std_list)
        X = self._remove_stopwords(X, self.stopword_list)
        X = self._apply_stemmer(X)
        X = self._untokenize_text(X)
        return X

if __name__ == "__main__":
    # predictor
    X_col = 'tweet_text'
    # classifier
    y_col = 'sentiment'
    prep = Preprocess()
    train_ds = pd.read_csv('Train.csv', delimiter=';')
    # update classifiers to nominal value
    train_ds[y_col] = train_ds[y_col].map({0: 'Negative', 1: 'Positive'})
    X_train = train_ds.loc[:, X_col].values
    y_train = train_ds.loc[:, y_col].values

    test_ds = pd.read_csv('Test.csv', delimiter=';')

    test_ds[y_col] = test_ds[y_col].map({0: 'Negative', 1: 'Positive'})
    X_test = test_ds.loc[:, X_col].values
    y_test = test_ds.loc[:, y_col].values

    X_train = prep.run(X_train)
    X_test = prep.run(X_test)

    dump((X_train, y_train, X_test, y_test), 'preprocessed.pkl')
