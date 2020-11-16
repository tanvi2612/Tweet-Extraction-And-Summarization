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
from preprocessing import clean

word_embeddings = {}
f = open('../../glove/archive/glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

stop_words = stopwords.words('english')

# tokenize received sentences


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
            v = sum([word_embeddings.get(w, np.zeros((100,)))
                    for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    return sentence_vectors

# computing similarity score for vectors


def normalise(A):
    lengths = (A**2).sum(axis=1, keepdims=True)**.5
    return A/lengths


def similarity_matrix(sentence_vectors):
    # sim_mat = np.zeros([len(sentence_vectors), len(sentence_vectors)])
    # print(len(sentence_vectors[0]))
    A = normalise(sentence_vectors)
    B = normalise(sentence_vectors)

    results = []

    rows_in_slice = 100

    slice_start = 0
    slice_end = slice_start + rows_in_slice

    while slice_end <= A.shape[0]:

        results.append(A[slice_start:slice_end].dot(B.T).max(axis=1))

        slice_start += rows_in_slice
        slice_end = slice_start + rows_in_slice

    result = np.concatenate(results)
    return result
    # for i in range(len(sentence_vectors)):
    #         for j in range(len(sentence_vectors)):
    #             if i != j:
    #                 sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(
    #                     1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]
    #     return sim_mat

# apply page rank algo

def apply_pagerank(sim_mat):
    nx_graph = nx.DiGraph(sim_mat)
    scores = nx.pagerank(nx_graph)
    return scores

# extract tweet summary based on page-rank algo
def summary_extraction(scores, sentences,k):
    ranked_sentences = sorted(((scores[i], s) for i,s in enumerate(sentences)), reverse=True)
    for i in range(k):
        if i >=len(ranked_sentences):
            break
        print(ranked_sentences[i][1])


tweets = []
with open("../dataset/raw-dataset/raw") as f:
    lines = f.readlines()
    hashtag_count = 0
    mention_count = 0
    punct_count = 0
    for l in lines:
        tweet = json.loads(l)
        # print(tweet.keys())
        text, m, h, p = clean(tweet["full_text"])
        hashtag_count += h
        mention_count += m
        punct_count += p
        # print(tweet.keys())
        tweets.append([tweet["id_str"], text])
        # break
# dataset = json_normalize(d["id"])
df = pd.DataFrame(data=tweets, columns=["ID", "Text"])
orig = len(df.index)
df = df.drop_duplicates('Text', keep='last')
unq = set()
df['Text'].str.lower().str.split().apply(unq.update)
tot = df['Text'].apply(lambda x: len(str(x).split()))
print("The total number of unique tweets considered are:", len(df.index))
print("The total number of duplicate tweets are:", orig - len(df.index))
print("-----Stats---------")
print("The Total Number of Words in the text are:", sum(tot))
print("The Number of Words in the longest Tweet:", max(tot))
print("The Number of Words in the shortest Tweet:", min(tot))
print("The Total Number of Unique Words in the text are:", len(unq))
print("The Total Number of Hashtags in the text are:", hashtag_count)
print("The Total Number of Mentions in the text are:", mention_count)
print("The Total Number of Puntuations in the text are:", punct_count)
# print(df.head)
# df.to_pickle("../dataset/data.pkl")
# tweets = df["Text"].tolist()
# print(len(tweets))
# # tweets = [tweet[1] for tweet in tweets]
# sentences = sentence_tokenize(tweets)
# print("done tokenize")
# clean_sentences = text_processing(sentences)
# print("done preprocessing")
# sentence_vectors = vector_representations(clean_sentences)
# sentence_vectors=np.array([np.array(xi) for xi in sentence_vectors])
# print("done vector rep", sentence_vectors.shape)
# sim_mat = similarity_matrix(sentence_vectors)
# print("done similarity matrix", sim_mat.shape)
# scores = apply_pagerank(sim_mat)
# print("done pagerank")
# summary_extraction(scores, sentences, 5)
