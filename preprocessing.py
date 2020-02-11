import pandas as pd
from nltk import word_tokenize


class Keyword:

    def __init__(self, keyword_col):
        self.keyword_col = pd.DataFrame(keyword_col)

    def fill_na(self):
        return self.keyword_col.fillna(" ")

    def replace(self, to_be_replaced, to_replace = " "):
        return self.keyword_col.str.replace(to_be_replaced, to_replace)


class Location:
    def __init__(self, location_col):
        self.location_col = pd.DataFrame(location_col)

    def fill_na(self):
        return self.location_col.fillna(" ")

    def preprocess_location(self):
        # remove weird characters
        # weird_char = re.compile(r"[^A-Za-z0-9]+")
        # location =  weird_char.sub(r"", location)

        # split the words into lists
        tokens = word_tokenize(self.location_col)

        # lowercase all words
        tokens = [word.lower() for word in tokens]

        # remove non-alphabet characters
        tokens = [word for word in tokens if word.isalpha()]

        # remove stop words
        tokens = [word for word in tokens if word not in stop_words]

        return " ".join(tokens)