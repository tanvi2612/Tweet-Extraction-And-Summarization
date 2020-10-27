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
from page_rank import get_sentences, sentence_tokenize,remove_stopwords,text_processing, vector_representations


MAX_TWEETS = 200
REQUIRED_TWEETS = 100

CONSUMER_TOKEN = "jHTLoXi1itNEIboVkkG5PlZlM"
CONSUMER_SECRET = "P4GsPqM4ms9amDLdHC69aKXk1gom7NH17apofBCDbMmp2uBgZ5"
ACCESS_TOKEN = "554538683-luV4dNLwBP33ifxIjdCRMljx2wAftj01RcRvjhrI"
ACCESS_TOKEN_SECRET = "kRoBfbr3HBnMKCcXVaPwGtIj8P2Mo2LmgXjw86JI2OOlf"
auth = tweepy.OAuthHandler(CONSUMER_TOKEN, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

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

def fetch_and_clean_tweets(hashtag,slang_dictionary):
	if not os.path.isdir(tweets_folder+hashtag):
		os.mkdir(tweets_folder+hashtag)
	hash_tag_folder = tweets_folder+hashtag
	count=0
	seen = []
	for tweet in tweepy.Cursor(api.search, q="#"+hashtag, lang="en", tweet_mode='extended').items(MAX_TWEETS):
		if 'retweeted_status' in dir(tweet):
			text=tweet.retweeted_status.full_text
		else:
			text=tweet.full_text
		
		if 'in_reply_to_status_id' in dir(tweet):
			if tweet.in_reply_to_status_id != None:
				continue

		try:
			if tweet.entities['media'][0]['type']=='photo':
				continue

			if tweet.entities['media'][0]['type']=='video':
				continue
		except:
			pass

		text = re.sub('&amp;','&',text)
		text = re.sub(r'^RT[\s]+', '', text, flags=re.MULTILINE) # removes RT
		text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) #remove link
		text = re.sub(r'[:]+', '', text, flags=re.MULTILINE)	
		text = re.sub(r'[^\x00-\x7F]+','', text)
		for wr in text:
			if wr in slang_dictionary:
				text= re.sub(r'\b'+wr+r'\b', slang_dictionary[wr],text)

		new_text = ""		
		for i in text.split(): # remove @ and #words, punctuataion
			if not i.startswith('@') and not i.startswith('#') and i not in string.punctuation:
				new_text+=i+" "	
		text = new_text			
		if len(text)>80:
			if text in seen:
				continue
			else:
				count+=1
				seen.append(text)
				write_to_file(tweets_folder+hashtag+"/"+str(count), text)
		if count>=REQUIRED_TWEETS:
			break

if not os.path.isdir("data/tweets"):
	os.mkdir("data/tweets")
slang_dictionary ={}
csvfile= open('Slang_Dictionary.csv','r')
reader = csv.reader(csvfile)
for row in reader :
	slang_dictionary[str(row[0])]= str(row[1])

while(1):
	print("Enter a hashtag without the hash: ")
	input_hashtag = input()
	fetch_and_clean_tweets(input_hashtag,slang_dictionary)
	sentences = get_sentences(input_hashtag)
	print("Received tweets..............")
	print(sentences[:5])
	sentences = sentence_tokenize(sentences)
	print("Tokenized sentences............")
	print(sentences[:5])
	clean_sentences = text_processing(sentences)
	print("After preprocessing...............")
	print(clean_sentences[:5])
	sentence_vectors = vector_representations(clean_sentences)
	print("Vectors Received.................")
	print(sentence_vectors:5)
	
	
