import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV


def vectorize(train_df, test_df):

    tfidf = TfidfVectorizer()
    tfidf.fit(train_df.preprocessed_text)
    train_tfidf = pd.DataFrame(tfidf.transform(train_df.preprocessed_text).toarray())
    test_tfidf = pd.DataFrame(tfidf.transform(test_df.preprocessed_text).toarray())

    return train_tfidf, test_tfidf


def classify(train_df, test_df):

    # not done here
    mnb_clf = MultinomialNB()   # alpha=3.295
    cnb_clf = ComplementNB()
    ridge_clf = RidgeClassifier()
    log_reg = LogisticRegression()
    sgd_clf = SGDClassifier()

    # clf_mnb.fit(train_set, train_df.target)
    # output_df["target"] = clf_mnb.predict(test_set)
    #
    # output_df.to_csv("submission.csv", index=False)
    # sgd_scores: [0.55185185 0.48120301 0.56277056 0.52066116 0.62250712]
    # mean score: 0.5478


def get_scores(classifier, train_tfidf, train_target):
    k_fold = KFold(5, shuffle=True, random_state=1).get_n_splits(train_tfidf.values)
    scores = cross_val_score(classifier, train_tfidf, train_target, cv=k_fold, scoring="f1")
    return scores
