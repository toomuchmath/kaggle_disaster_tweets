import pandas as pd
from preview import preview
from preprocessing import preprocess
from classify import compare_cv_scores, compare_results, predict_test


# load data
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")
output_df = pd.read_csv("../data/sample_submission.csv")

preview(train_df, test_df)
train_df, test_df = preprocess(train_df, test_df)

compare_cv_scores(train_df)
# output
# mnb: [0.52946679 0.5590609  0.6166552  0.5909465  0.72292546], mean: 0.6038
# cnb: [0.56       0.56254319 0.63559871 0.60644148 0.72048032], mean: 0.6170
# ridge: [0.53018868 0.49847561 0.58041458 0.60130719 0.67663043], mean: 0.5774
# log_reg: [0.52958293 0.50316456 0.59035277 0.5470852  0.69811321], mean: 0.5737
# sgd: [0.55197792 0.50408922 0.58292852 0.57004049 0.681074  ], mean: 0.5780
# rf: [0.56628788 0.45537948 0.54453295 0.43361987 0.62568952], mean: 0.5251

compare_results(train_df)       # I proceeded with mnb and cnb only at this stage
predict_test(train_df, test_df, output_df)      # I picked mnb for final prediction
