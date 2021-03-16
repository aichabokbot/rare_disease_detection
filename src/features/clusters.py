import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
plt.style.use('ggplot')
from scipy.spatial.distance import cdist
nltk.download('punkt')
nltk.download('stopwords')

def first_data(data):
    """
    Get right data for text processing
    """
    #selecting only anamnestic data for clustering
    data=data[data.TYP=='A']
    data=data.groupby(['PATIENT_HASH'])['TEXT'].apply(','.join).reset_index()
    #adding word count for themedical texts
    data['notes_word_count']=data['TEXT'].apply(lambda x:len(x.strip().split()))
    return data
def clean_text(text, for_embedding=False):
    """
    - remove any html tags (< /br> often found)
    - Keep only ASCII + European Chars and whitespace, no digits
    - remove single letter chars
    - convert all whitespaces (tabs etc.) to single wspace
    if not for embedding (but e.g. tdf-idf):
    - all lowercase
    - remove stopwords, punctuation and stemm
    """

    stemmer = SnowballStemmer("german")
    stop_words = stopwords.words("german")
    wods=['hypercholesterinaemi','hypercholesterinami','hypercholesterinamie','hypercholesterinami hypercholesterinami','hypercholesterinamie','hypercholesterinami hypercholesterinami']
    stop_words.extend(wods)
    
    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
    RE_TAGS = re.compile(r"<[^>]+>")
    RE_ASCII = re.compile(r"[^A-Za-zÀ-ž ]", re.IGNORECASE)
    RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE)
    if for_embedding:
        # Keep punctuation
        RE_ASCII = re.compile(r"[^A-Za-zÀ-ž,.!? ]", re.IGNORECASE)
        RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž,.!?]\b", re.IGNORECASE)

    text = re.sub(RE_TAGS, " ", text)
    text = re.sub(RE_SINGLECHAR, " ", text)
    text = re.sub(RE_WSPACE, " ", text)

    word_tokens = word_tokenize(text)
    words_tokens_lower = [word.lower() for word in word_tokens]

    if for_embedding:
        # no stemming, lowering and punctuation / stop words removal
        words_filtered = word_tokens
    else:
        words_filtered = [
            stemmer.stem(word) for word in words_tokens_lower if word not in stop_words
        ]

    text_clean = " ".join(words_filtered)
    
    return text_clean


def vec(df):
    """
    Applying the function and gettin the vectorized dataframe X
    """
    df["comment_clean"] = df["TEXT"].map(
        lambda x: clean_text(x, for_embedding=False) if isinstance(x, str) else x
    )
    df["comment_clean"]=df.comment_clean.str.split().map(lambda x:" ".join(s for s in x if len(s) > 8))

    """
    Compute unique word vector with frequencies
    exclude very uncommon (<10 obsv.) and common (>=30%) words
    use pairs of two words (ngram)
    """
    vectorizer = TfidfVectorizer(
        analyzer="word", max_df=0.3, min_df=10, ngram_range=(1, 2), norm="l2"
    )
    X=vectorizer.fit_transform(df["comment_clean"])
    return X

def get_cluster(X,df):
    """
    Get dataframe with clusters
    """
    number_of_clusters = 15

    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    model = KMeans(n_clusters=number_of_clusters, 
                   init='k-means++', 
                   max_iter=100, # Maximum number of iterations of the k-means algorithm for a single run.
                   n_init=1)  # Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.

    model.fit(X)
    df['clusters'] = pd.Series(model.predict(X), index=df.index)
    return df




    data = pd.read_csv('/home/jovyan/amedes_challenge/data/interim/data_extract_preprocessed.csv')    
    df=first_data(data)
    X=vec(df)
    clusters=get_cluster(X,df)
    
