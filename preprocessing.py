import pandas as pd
import re
import string
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer


def preprocess(train_df, test_df):

    train_rows = train_df.shape[0]
    test_rows = test_df.shape[0]

    # combine train and test dfs for easy preprocessing (I will split them up later)
    full_df = pd.concat([train_df, test_df])

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

    full_df["cleaned_text"] = clean_text(full_df.text)

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

    # preprocess text
    full_df["preprocessed_text"] = full_df.cleaned_text.apply(lambda text: preprocess_text(text))

    # preprocess location
    full_df.location = full_df.location.fillna(" ")
    full_df["preprocessed_location"] = full_df.location.apply(lambda location: preprocess_location(location))

    # preprocess keyword
    full_df.keyword = full_df.keyword.fillna(" ")
    full_df.keyword = full_df.keyword.str.replace("%20", " ")

    # combine preprocessed keyword, location and text into one column
    full_df["preprocessed_text"] = full_df["keyword"] + " " + full_df["preprocessed_location"] + " " \
                                   + full_df["preprocessed_text"]

    # split into train and test sets
    train_preprocessed = full_df.iloc[:train_rows, :]
    test_preprocessed = full_df.iloc[train_rows:, :]

    assert train_preprocessed.shape[0] == train_rows
    assert test_preprocessed.shape[0] == test_rows

    return train_preprocessed, test_preprocessed
