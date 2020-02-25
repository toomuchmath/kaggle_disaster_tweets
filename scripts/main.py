import pandas as pd
from preview import preview
from preprocessing import preprocess
from classify import compare_cv_scores, compare_results, predict_test


# load data
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")
output_df = pd.read_csv("../data/sample_submission.csv")

preview(train_df,test_df)
train_df, test_df = preprocess(train_df, test_df)

compare_cv_scores(train_df)
compare_results(train_df)
predict_test(train_df, test_df, output_df)