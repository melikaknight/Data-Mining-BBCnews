#!/usr/bin/env python
# coding: utf-8

# # بسم الله الرحمن الرحیم

# #### Abstract 
# The whole proccess is like this: 
# + First, we read all the data, and do preproccessing
# - By using [word2vec](https://medium.com/data-science-group-iitr/word-embedding-2d05d270b285), we apply an embedding to all words of the text, and by summing all words of the text, created embedded vector of the text
# * Then apply a [Kmeans]() algorithm on these vectors, and cluster the news
# + We also use [PCA](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c) to visuallize the data
# ![alt text](abs.png "Title")
# 

# In[16]:


import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn.decomposition import PCA,NMF
import matplotlib.pyplot as plt
from nltk import  stem, word_tokenize,download
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandasql as ps
from nltk import  stem, word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import codecs
import re


# You might need to download some parts of NLTK too ...

# In[17]:


download('punkt')
download('stopwords')


# Then we read the news, and store them in a pandas DataFrame along side their subjects.

# In[18]:


subjects=['tech','sport','politics','entertainment','business']
news=pd.DataFrame(columns=['subject','text'])
n_samples=500
for i in range(1,n_samples):
    for sub in subjects:
        try:
            file_name='bbc-raw/'+sub+'/'+str(i).zfill(3)+'.txt'
            #print(file_name)
            text_file=open(file_name)
            n={'subject':sub,'text':text_file.read()}
            news=news.append(n, ignore_index=True )
        except:
            pass
news.head()


# In[ ]:





# Now we need to cleanse then preprocess the text. for cleansing:
# +  We define acceptable characters, and then using regex, we get rid of anything that is not 
# ```        junk_chars_regex=r'[^a-zA-Z0-9\u0621-\u06CC\u0698\u067E\u0686\u06AF \u200c]'```
# 
# - Then find any possible URLs, and remove them too
# ```        url_regex = r"""(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’]))"""```
# 
# 
# + Emails too
# ``` RFC_5322_COMPLIANT_EMAIL_REGEX = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"```
# 
# + The last but not the least, we remove stop words (things like 'and', 'or' etc ), using NLTK's list
# ```        self.stopwords=set(stopwords.words('english'))```
# 
# And for preproccessing, we do two steps. by `stem` we map all complex phrases to simple phrase, like gaming, games, gamer all get mapped to game, and then we break sentences to words by `wordTokenizing`
# 

# In[19]:


compile_patterns = lambda patterns: [(re.compile(pattern), repl) for pattern, repl in patterns]

class PreprocessDescription(object):
    def __init__(self, lemmatizer_params, tokenizer_params):
        self.lemmatizer = stem.PorterStemmer(**lemmatizer_params)
        
        junk_chars_regex=r'[^a-zA-Z0-9\u0621-\u06CC\u0698\u067E\u0686\u06AF \u200c]'
        url_regex = r"""(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’]))"""
        RFC_5322_COMPLIANT_EMAIL_REGEX = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
        
        remove_url=(url_regex, ' ')
        remove_email=(RFC_5322_COMPLIANT_EMAIL_REGEX, ' ')
        remove_junk_characters=(junk_chars_regex, ' ')
        self.compiled_patterns_before = compile_patterns([remove_url,remove_email])
        self.compiled_patterns_after = compile_patterns([remove_junk_characters])
        self.stopwords=set(stopwords.words('english'))
        
    def preprocess(self, text):
        # html parser
        soup = BeautifulSoup(text, 'html.parser')
        for br in soup.find_all("br"):
            br.replace_with("\n ")
        text = soup.getText()
        text = text.lower()
        for pattern, repl in self.compiled_patterns_before:
            text = pattern.sub(repl, text)
        text = re.sub(r'[\u200c\s]*\s[\s\u200c]*', ' ', text)
        text = re.sub(r'[\u200c]+', '\u200c', text)
        
        for pattern, repl in self.compiled_patterns_after:
            text = pattern.sub(repl, text)
            
        ## tokenized_words
        tokenized_words = word_tokenize(text)
        ## lemmatized_data
        lemmatized_words = [self.lemmatizer.stem(word) for word in tokenized_words]
        lemmatized_witout_stopwords = [word for word in lemmatized_words if word not in self.stopwords]
        return lemmatized_witout_stopwords


# All English stopwords:

# In[20]:


stop=set(stopwords.words('english'))
print(stop)


# In[21]:


preprocessDesc = PreprocessDescription(lemmatizer_params={},tokenizer_params={})


# Now we apply preprocessing on all the news, and measure the time needed

# In[23]:


import time
start = time.time()
preprocessed_text = []

for index, row in news.iterrows():
    res = preprocessDesc.preprocess(row['text'])
    preprocessed_text.append(" ".join(res))
    
news['preprocessed_text']=preprocessed_text
print(time.time() - start)


# Now data looks like this ...

# In[24]:


news.head()


# Using word2vec model of gensim library, we vectorize all words of the text

# In[25]:


DIMENSION = 100
words=[]
for index,row in news.iterrows():
    words.append(row['preprocessed_text'].split())

w2v_model = Word2Vec(words,sg=1,iter=10,size=DIMENSION)


# and sum them up to created embedded vector of each news

# In[26]:


news['w2v_vector'] = pd.Series(np.zeros((news.shape[0])), index=news.index)
b=[]

for index, row in news.iterrows():
    a=np.zeros(DIMENSION)
    for word in row['preprocessed_text'].split():
        try:
            a = a  + w2v_model.wv[word]
        except:
            pass
    b.append(a)

news['w2v_vector']=b
news.head()


# A stop, it's better to check if our embedding is sufficient for clustering. We use PCA to visuallize the data and see if there is enought seperation between them.

# In[27]:


b_normal=normalize(b)
pca = PCA(n_components=2,whiten=True)
c_pca = pca.fit_transform(pd.DataFrame(b_normal))


# In[28]:


def show_original(news):
    fig,ax = plt.subplots(1,1,figsize=(7,4))
    for index, row in news.iterrows():
        if row['subject']=='sport':
            plt.scatter(c_pca[index][0],c_pca[index][1],color='red')
        elif row['subject']=='politics':
            plt.scatter(c_pca[index][0],c_pca[index][1],color='blue')
        elif row['subject']=='tech':
            plt.scatter(c_pca[index][0],c_pca[index][1],color='green')
        elif row['subject']=='entertainment':
            plt.scatter(c_pca[index][0],c_pca[index][1],color='yellow')
        elif row['subject']=='business':
            plt.scatter(c_pca[index][0],c_pca[index][1],color='black')
    plt.legend(['tech', 'sport', 'politics', 'entertainment', 'business'])
    plt.title('original data')
    plt.show()
show_original(news)


# Seems good to me!
# 
# Then, we use cosine similarity between these vectors ti see if cosine similar vectors of a text are clost to it.

# In[29]:


def get_w2v_top_similar_docs(data,text_vectors, index, k):
    data_res= data
    data_res['cosine'] = pd.Series(np.zeros(data_res.shape[0]), index=data_res.index)

    #package = data_res.loc[data_res['package_name']==package_name]
    comparing_vec = text_vectors[index]

    for index, row in data_res.iterrows():
        data_res.loc[index,'cosine']=(cosine_similarity(comparing_vec.reshape(1, -1),row['w2v_vector'].reshape(1, -1))[0])
#        pass-
    return data_res.nlargest(k, 'cosine')


# 10 similar news of 2nd sports news are pretty similar!

# In[30]:


examples_df = get_w2v_top_similar_docs(news, b, index=1, k=5)
examples_df.head(10)


# And at last, using KMeans, we do a clustering with 5 possible labels, and again visualize the data

# In[31]:


km = KMeans(n_clusters=5)
km.fit(b_normal)
#print(km.cluster_centers_)


# In[32]:


def show_clustered(data,km):
    fig,ax = plt.subplots(1,1,figsize=(7,4))
    c=[]
    pca = PCA(n_components=2,whiten=True)
    for i in range(len(data)):
        #print(data[i])
        c.append(km.predict(data[i].reshape(1,-1))[0])

    X = pca.fit_transform(pd.DataFrame(data))
    #print(X.shape)
    for index in range(X.shape[0]):
        if c[index] ==0:
            plt.scatter(X[index][0],X[index][1],color='red')
        elif c[index] ==1:
            plt.scatter(X[index][0],X[index][1],color='blue')
        elif c[index] ==2:
            plt.scatter(X[index][0],X[index][1],color='green')
        elif c[index] ==3:
            plt.scatter(X[index][0],X[index][1],color='yellow')
        elif c[index] ==4:
            plt.scatter(X[index][0],X[index][1],color='black')
    plt.title('clustered data')
    plt.show()
    return c


# In[34]:


#print(b)
c=show_clustered(b_normal,km)
show_original(news)
news['label']=c[:]
#for i in range(len(c)):


# In[35]:


news.head(10)


# And see the precision of our clustering, by seeing how many of news with same category got the same label. results are prety solid! Top five labels are assigned to different subjects!

# In[36]:


query="""SELECT subject,label,count(*) c FROM news group by subject,label order by c DESC"""

print(ps.sqldf(query, locals()))


# In[ ]:




