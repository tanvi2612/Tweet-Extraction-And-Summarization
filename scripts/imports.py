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

from textblob import TextBlob
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import matplotlib.pyplot as plt
from matplotlib import rc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import seaborn as sns

