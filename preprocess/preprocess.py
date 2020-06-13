#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
nltk.download("stopwords")
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

import nltk
import keras
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


punctuations="?:!.,;"


# In[ ]:


wordnet_lemmatizer = WordNetLemmatizer()


# In[3]:


def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
      if word in punctuations:
          token_words.remove(word)
    for word in token_words:
        stem_sentence.append(wordnet_lemmatizer.lemmatize(word, 'v'))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


# In[ ]:




