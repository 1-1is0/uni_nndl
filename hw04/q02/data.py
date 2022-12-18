# %%
import pandas as pd
import numpy as np
import re

# %%
df = pd.read_csv("./data/FA-KES-Dataset.csv",
                 encoding="unicode_escape", index_col="unit_id")
# %%
df.head()
# %%
def match_ip(text):
    IP_REGEX = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    cond = re.findall(IP_REGEX, text)
    if cond:
        print(cond)
    t = re.sub(IP_REGEX, "", text)
    return t

def match_url(text):
    URL_REGEX = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
    URL_REGEX = r"""((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"""
    cond = re.findall(URL_REGEX, text)
    if cond:
        print(cond)
    t = re.sub(URL_REGEX, "", text)
    return t
    

df.article_content = df.article_content.apply(match_ip)
df.article_content = df.article_content.apply(match_url)

# %%
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def pre(text):
    stop_words = set(stopwords.words("english"))
    # sentence_tokens = sent_tokenize()
    # TODO should remove punctiations too ?
    # TODO should all of them be in a array [], or [[], []]
    terms = []
    for sentence in sent_tokenize(text):
        for word in word_tokenize(sentence):
            if word.lower() not in stop_words:
                terms.append(stemmer.stem(word))
    print(terms)

df.article_content.head(1).apply(pre)