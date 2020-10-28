import tweepy
import re
import string
import os
import csv
from operator import add
from scipy import spatial
import glob
import sys
import numpy as np
from scipy.spatial import distance
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import random
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pandas as pd
import re
import networkx as nx
from nltk.corpus import stopwords
from page_rank import get_sentences, sentence_tokenize,remove_stopwords,text_processing, vector_representations, similarity_matrix,apply_pagerank,summary_extraction
from scraper import get_autorization, fetch_and_clean_tweets

MAX_TWEETS = 20000
REQUIRED_TWEETS = 5000


tweets_folder = "data/tweets/"

word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
stop_words = stopwords.words('english')


def write_to_file(filename, tweet):
	with open(filename, "w") as f: 
		f.write(tweet) 

if not os.path.isdir("data/tweets"):
	os.mkdir("data/tweets")
slang_dictionary ={}
csvfile= open('Slang_Dictionary.csv','r')
reader = csv.reader(csvfile)
for row in reader :
	slang_dictionary[str(row[0])]= str(row[1])
api = get_autorization()
while(1):
	print("Enter a hashtag without the hash: ")
	input_hashtag = input()
	fetch_and_clean_tweets(api, tweets_folder, input_hashtag, slang_dictionary, MAX_TWEETS)
	sentences = get_sentences(input_hashtag)
	print("Received tweets..............")
	# print(sentences[:5])
	sentences = sentence_tokenize(sentences)
	# print("Tokenized sentences............")
	# print(sentences[:5])
	clean_sentences = text_processing(sentences)
	print("Preprocessing data...............")
	# print(clean_sentences[:5])
	sentence_vectors = vector_representations(clean_sentences)
	print("Vectors Representations Calculated.................")
	# print(sentence_vectors[:5])
	sim_mat = similarity_matrix(sentence_vectors)
	print("Similarity matrix calculated.................")
	scores = apply_pagerank(sim_mat)
	print("page rank scores..............")
	print("SUMMARY:")
	summary_extraction(scores,sentences,10)
	
