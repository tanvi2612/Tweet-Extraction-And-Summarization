import nltk
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import glob
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import json

word_embeddings = {}
f = open('../../glove/archive/glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

stop_words = stopwords.words('english')




# read tweets corresponding to the tag
def get_sentences(tag):
    files = glob.glob("data/tweets/"+tag+"/*")
    sentences = []
    for file in files:
        fd = open(file,"r")
        for line in fd:
            sentences.append(line)
    return sentences


#tokenize received sentences
def sentence_tokenize(tweets):
    sentences = []
    for tweet in tweets:
        sentences.append(sent_tokenize(tweet))
    sentences = [y for x in sentences for y in x]
    return sentences


def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


# preprocessing --> lowercase, stopwords removal
def text_processing(sentences):
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = [s.lower() for s in clean_sentences]
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    return clean_sentences


# get vector representations using glove embedding
def vector_representations(clean_sentences):
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    return sentence_vectors

#computing similarity score for vectors
def similarity_matrix(sentence_vectors):
    sim_mat = np.zeros([len(sentence_vectors), len(sentence_vectors)])
    for i in range(len(sentence_vectors)):
        for j in range(len(sentence_vectors)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    return sim_mat

#apply page rank algo
def apply_pagerank(sim_mat):
    nx_graph = nx.DiGraph(sim_mat)
    scores = nx.pagerank(nx_graph)
    return scores

#extract tweet summary based on page-rank algo
def summary_extraction(scores,sentences,k):
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    for i in range(k):
        if i>=len(ranked_sentences):
            break
        print(ranked_sentences[i][1])


tweets = []
with open("../dataset/IRE Project-20201029T070205Z-001/raw") as f:
    lines = f.readlines()
    for l in lines:
        tweet = json.loads(l)
        # print(tweet.keys())
        tweets.append([tweet["id_str"], tweet["full_text"]])
# dataset = json_normalize(d["id"])
df = pd.DataFrame(data=tweets, columns=["ID", "Text"])
# print(df.head)
df.to_pickle("../dataset/data.pkl")