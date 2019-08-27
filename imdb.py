import csv
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS 
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
import matplotlib.pyplot as plt 
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("C:/Users/irmri/OneDrive/Desktop/IMDB Dataset.csv") # read csv
df = df.head(200)

df['review'] = df['review'].str.lower()  # convert all the words to lowercase
df['words'] = df['review'].str.replace('[^\w\s]','') # remove punctuation marks
df['words'] = df['words'].apply(nltk.word_tokenize)  # split the sentence into tokens


stop_words = set(stopwords.words('english'))

def remove_stops(row):
    my_list = row['words']
    meaningful_words = [w for w in my_list if not w in stop_words]
    return (meaningful_words)

df['target'] = df.apply(remove_stops, axis=1) 


stemming = PorterStemmer()
def stem_list(row):
    my_list = row['target']
    stemmed_words = [stemming.stem(s) for s in my_list]
    return (stemmed_words)
    
df['stemmed'] = df.apply(stem_list, axis=1)



train = df.sample(frac=0.8, random_state=200) # train set
test = df.drop(train.index) # test set

train_pos = train[train.sentiment == "positive"]  #training set with positive sentiment
train_neg = train[train.sentiment == "negative"]  # training set with negative sentiment




def show_wordcloud(train_examples, title = None):
    wordcloud = WordCloud(
        background_color='black',
       # stopwords=stopwords,
        max_words=10,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(train_examples))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(train_pos['review'])
show_wordcloud(train_neg['review'])








