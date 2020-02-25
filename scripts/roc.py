import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test


def split_fit_predict(X, y, clf):

    X_train, X_test, y_train, y_test = split(X, y)

    # classify
    clf.fit(X_train, y_train)

    # predict
    y_pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    probs = probs[:, 1]

    return y_test, y_pred, probs


def get_roc(y_test, probs):

    auc = roc_auc_score(y_test, probs)
    fpr, tpr, thresholds = roc_curve(y_test, probs)

    return fpr, tpr, auc


def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='best')
    plt.show()
