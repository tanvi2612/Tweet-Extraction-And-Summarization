#!/usr/bin/python3
import configparser
import tweepy
import pandas as pd
import sys

def get_tweets(api, outfile, query="", count=5e4):
    tweets = api.search(query=query, count=count)
    frames = [pd.json_normalize(i._json) for i in tweets]
    df = pd.concat(frames)
    df.to_pickle(outfile)

if __name__=="__main__":
    cn = configparser.ConfigParser()
    cn.read('../.config')
    keys = cn['KEYS']

    auth = tweepy.OAuthHandler(keys['API_KEY'], keys['API_SECRET_KEY'])
    auth.set_access_token(keys['ACCESS_TOKEN'], keys['ACCESS_TOKEN_SECRET'])
    api = tweepy.API(auth)

    query = sys.argv[1]
    count = sys.argv[2]
    get_tweets(api, "../dataset/"+query, query, int(count))

