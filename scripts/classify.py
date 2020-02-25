import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from roc import split_fit_predict, get_roc, plot_roc_curve


def vectorize(train_df, test_df):

    tfidf = TfidfVectorizer()
    tfidf.fit(train_df.preprocessed_text)
    train_tfidf = pd.DataFrame(tfidf.transform(train_df.preprocessed_text).toarray())
    test_tfidf = pd.DataFrame(tfidf.transform(test_df.preprocessed_text).toarray())

    return train_tfidf, test_tfidf


def clf_pipeline(classifier):

    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', classifier)])

    return clf


def instantiate_clf():

    mnb_clf = clf_pipeline(MultinomialNB(alpha=0.75))
    cnb_clf = clf_pipeline(ComplementNB(alpha=2.5))
    ridge_clf = clf_pipeline(RidgeClassifier())
    log_reg = clf_pipeline(LogisticRegression(solver='saga', C=2))
    sgd_clf = clf_pipeline(SGDClassifier())
    rf_clf = clf_pipeline(RandomForestClassifier())

    classifiers = {'mnb': mnb_clf,
                   'cnb': cnb_clf,
                   'ridge': ridge_clf,
                   'log_reg': log_reg,
                   'sgd': sgd_clf,
                   'rf': rf_clf}

    return classifiers


def get_scores(classifier, X, y):
    k_fold = KFold(5, shuffle=True, random_state=1).get_n_splits(X.values)
    scores = cross_val_score(classifier, X, y, cv=k_fold, scoring="f1")
    return scores


def get_best_params(params, classifier, X, y):

    search_clf = RandomizedSearchCV(classifier, params, random_state=1)
    search = search_clf.fit(X, y)

    return search.best_params_  # return a dict of best parameters in 'distributions' arg


def compare_cv_scores(train_df):

    classifiers = instantiate_clf()

    for clf_name, clf in classifiers.items():
        clf_scores = get_scores(clf, train_df.preprocessed_text, train_df.target)
        print("{}: {}, mean: {:.4f}".format(clf_name, clf_scores, clf_scores.mean()))


def compare_results(train_df):

    mnb_clf = clf_pipeline(MultinomialNB(alpha=0.75))
    cnb_clf = clf_pipeline(ComplementNB(alpha=2.5))

    classifiers = {'mnb': mnb_clf,
                   'cnb': cnb_clf}

    for clf_name, clf in classifiers.items():
        y_test, y_pred, probs = split_fit_predict(train_df.preprocessed_text, train_df.target, clf)

        if clf_name != 'ridge':
            fpr, tpr, auc = get_roc(y_test, probs)
            print("{} AUC: {:.3f}".format(clf_name, auc))

            # using roc because there isn't class imbalance
            plot_roc_curve(fpr, tpr)

        matrix = confusion_matrix(y_test, y_pred)
        print(matrix)

        report = classification_report(y_test, y_pred)
        print(report)


def predict_test(train_df, test_df, output_df):

    mnb_clf = clf_pipeline(MultinomialNB(alpha=0.75))
    mnb_clf.fit(train_df.preprocessed_text, train_df.target)
    output_df["target"] = pd.DataFrame(mnb_clf.predict(test_df.preprocessed_text).astype(int))
    output_df.to_csv("../data/submission.csv", index=False)
