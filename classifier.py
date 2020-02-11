# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import string
from nltk.corpus import stopwords, wordnet
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag, word_tokenize
from preprocessing import Keyword

# load data

train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
output_df = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

# overview of data

print(train_df.info())
print('-'*40)
print(test_df.info())
print('-'*40)
print(train_df.isnull().sum())

# class balance
train_df.target.value_counts().plot(kind='bar')


# cleaning text

def remove_url(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)


def remove_html(text):
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r"", text)


def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


def clean_text(text_vector):
    text_vector = text_vector.apply(lambda text: remove_url(text)).apply(lambda text: remove_html(text)) \
        .apply(lambda text: remove_emoji(text)).apply(lambda text: remove_punct(text))
    return text_vector


train_df["cleaned_text"] = clean_text(train_df.text)
test_df["cleaned_text"] = clean_text(test_df.text)

# pre-processing text

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()


def get_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    else:
        return None


def lemmatize_tokens(tokens):
    # to collect all the lemmatized tokens
    lemmatized = []

    # pos tagging each token/word
    tokens_tags = pos_tag(tokens)

    for token_tag_tuple in pos_tag(tokens):

        # convert into wordnet tag for lemmatization
        wordnet_tag = get_wordnet_tag(token_tag_tuple[1])

        if wordnet_tag is not None:
            lemmatized.append(lemmatizer.lemmatize(token_tag_tuple[0], wordnet_tag))
        else:
            lemmatized.append(token_tag_tuple[0])

    return lemmatized


def preprocess_text(text):
    # split the words into lists
    tokens = word_tokenize(text)

    # lowercase all words
    tokens = [word.lower() for word in tokens]

    # remove non-alphabet characters
    tokens = [word for word in tokens if word.isalpha()]

    # remove stop words
    tokens = [word for word in tokens if word not in stop_words]

    # apply lemmatizer
    lemmatized = lemmatize_tokens(tokens)

    # apply stem
    stemmed = [porter.stem(word) for word in lemmatized]

    return " ".join(stemmed)


def preprocess_location(location):
    # remove weird characters
    # weird_char = re.compile(r"[^A-Za-z0-9]+")
    # location =  weird_char.sub(r"", location)

    # split the words into lists
    tokens = word_tokenize(location)

    # lowercase all words
    tokens = [word.lower() for word in tokens]

    # remove non-alphabet characters
    tokens = [word for word in tokens if word.isalpha()]

    # remove stop words
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


train_df["preprocessed_text"] = train_df.cleaned_text.apply(lambda text:preprocess_text(text))
test_df["preprocessed_text"] = test_df.cleaned_text.apply(lambda text:preprocess_text(text))

train_df.location = train_df.location.fillna(" ")
test_df.location = test_df.location.fillna(" ")
train_df["preprocessed_location"] = train_df.location.apply(lambda location:preprocess_location(location))
test_df["preprocessed_location"] = test_df.location.apply(lambda location:preprocess_location(location))

# preprocess keyword

"""
keyword_col = Keyword(train_df.keyword).preprocess()
"""

train_df.keyword = train_df.keyword.fillna(" ")
train_df.keyword = train_df.keyword.str.replace("%20", " ")

test_df.keyword = test_df.keyword.fillna(" ")
test_df.keyword = test_df.keyword.str.replace("%20", " ")

train_df["preprocessed_text"] = train_df["keyword"] + " " + train_df["preprocessed_location"] + " " + train_df["preprocessed_text"]
test_df["preprocessed_text"] = test_df["keyword"] + " " + test_df["preprocessed_location"] + " " + test_df["preprocessed_text"]

full_text_df = pd.concat([train_df["preprocessed_text"], test_df["preprocessed_text"]])

tfidf = TfidfVectorizer()
tfidf.fit(full_text_df)
train_set = tfidf.transform(train_df.preprocessed_text)
test_set = tfidf.transform(test_df.preprocessed_text)

clf_mnb = MultinomialNB(alpha=3.295)
clf_mnb.fit(train_set, train_df.target)
output_df["target"] = clf_mnb.predict(test_set)

output_df.to_csv("submission.csv", index=False)