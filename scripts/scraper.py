#!/usr/bin/python3
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

REQUIRED_TWEETS = 100

def write_to_file(filename, tweet):
	with open(filename, "w") as f: 
		f.write(tweet)

def fetch_and_clean_tweets(api, tweets_folder, hashtag, slang_dictionary, MAX_TWEETS=5e4):
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

# def get_tweets(api, outfile, query="", count=5e4):
#     tweets = api.search(query=query, count=count)
#     frames = [pd.json_normalize(i._json) for i in tweets]
#     df = pd.concat(frames)
#     df.to_pickle(outfile)

# if __name__=="__main__":
#     cn = configparser.ConfigParser()
#     cn.read('../.config')
#     keys = cn['KEYS']

def get_autorization():
  keys = dict()
  keys['API_KEY'] = "jHTLoXi1itNEIboVkkG5PlZlM"
  keys['API_SECRET_KEY'] = "P4GsPqM4ms9amDLdHC69aKXk1gom7NH17apofBCDbMmp2uBgZ5"
  keys['ACCESS_TOKEN'] = "554538683-luV4dNLwBP33ifxIjdCRMljx2wAftj01RcRvjhrI"
  keys['ACCESS_TOKEN_SECRET'] = "kRoBfbr3HBnMKCcXVaPwGtIj8P2Mo2LmgXjw86JI2OOlf"

  auth = tweepy.OAuthHandler(keys['API_KEY'], keys['API_SECRET_KEY'])
  auth.set_access_token(keys['ACCESS_TOKEN'], keys['ACCESS_TOKEN_SECRET'])
  api = tweepy.API(auth)
  return api

