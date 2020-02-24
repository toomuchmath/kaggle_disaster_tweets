import pandas as pd
from preview import preview
from preprocessing import preprocess
from classifying import vectorize, classify, get_scores
from sklearn.naive_bayes import ComplementNB

# load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
output_df = pd.read_csv("data/sample_submission.csv")

preview(train_df,test_df)
train_df, test_df = preprocess(train_df, test_df)

train_tfidf, test_tfidf = vectorize(train_df, test_df)

cnb = ComplementNB()
cnb_scores = get_scores(cnb, train_tfidf, train_df.target)
print("{}: {}, mean: {:.4f}".format("cnb_scores", cnb_scores, cnb_scores.mean()))