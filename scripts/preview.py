import matplotlib.pyplot as plt
import seaborn as sns


def preview(train_df, test_df):

    # overview of data
    train_df.info()
    print('-' * 40)
    test_df.info()
    print('-' * 40)
    print("Missing data count in train_df")
    print(train_df.isnull().sum())

    # class balance
    sns.countplot(x=train_df["target"])
    plt.ylabel("count")
    plt.title("target count")
    plt.show()
