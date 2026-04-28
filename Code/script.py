from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_excel('./Data/online_retail_II.xlsx')

print(data.info())