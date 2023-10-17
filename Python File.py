#!/usr/bin/env python
# coding: utf-8

# Sentiment Analysis 
# - VADER (Valence Aware Dictionary and Sentiment Reasoner) - Bag of Words
# - Roberta pretrained Model 
# - Huggingface pipeline

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
nltk.download('punkt')


# In[2]:


#plt.style.use('ggplot')


# In[3]:


reddit_df = pd.read_csv("reddit_data.csv")
twt_df = pd.read_csv("twitter_data.csv")


# In[4]:


reddit_df = (reddit_df.head(500))
reddit_df 
#print(reddit_df.shape)


# In[5]:


reddit_df.head()


# In[6]:


ax = reddit_df['category'].value_counts().sort_index().plot(kind='bar',
                                                      title='Count of Reviews',
                                                      figsize=(10,5))
#category > scores vid
ax.set_xlabel('Category') 
#category > review stars
plt.show()


# higher level of 1's depict positive reviews

# In[7]:


# Basic nltk
ex = reddit_df['clean_comment'][50]
print(ex)


# In[8]:


tokens = nltk.word_tokenize(ex)
tokens[:10]


# In[9]:


#declares parts of speech POS
nltk.download('averaged_perceptron_tagger')
tagged = nltk.pos_tag(tokens) 
tagged[:10]


# In[10]:


#groups the tokens into chunks of text
nltk.download('words')
nltk.download('maxent_ne_chunker')
entities = nltk.chunk.ne_chunk(tagged[:10])
entities.pprint()


# #VADER SENTIMENT SCORING (valence aware dictionary sentiment reasoner)
# nltk's Sentiment Intensity analyzer to get neg/neu/pos scores of the text
# >>uses bag of words approach

# In[11]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()


# In[12]:


sia.polarity_scores('I am feeling a lot bettter')


# In[13]:


sia.polarity_scores("this day just keeps getting worst")


# In[14]:


sia.polarity_scores(ex)


# In[15]:


reddit_df['Id'] = reddit_df.reset_index().index + 1


# In[16]:


list(reddit_df.columns)


# In[17]:


reddit_df['clean_comment'] = reddit_df['clean_comment'].astype(str)


# In[18]:


#runs the polarity score over the complete dataset
dic = {}
for i, row in tqdm(reddit_df.iterrows(), total=len(reddit_df)):
    text = row['clean_comment']
    myid = row['Id']
    dic[myid] = sia.polarity_scores(text)


# In[19]:


pd.DataFrame(dic).T
#stores the polarity scores in a pandas dataframe


# In[20]:


vaders = pd.DataFrame(dic).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(reddit_df, how='left')
#T flips horizontally
#merges the polarity scores with the original dataset


# In[21]:


vaders


# In[22]:


sns.barplot(data = vaders, x='category', y='compound')
ax.set_title('Compound Score by Reddit Star Reviews', loc='center')
plt.tight_layout()
plt.show()


# the reviews gets less negative as the rating goes higher

# In[23]:


fig, axs = plt.subplots(1,3, figsize=(12,3)) 
#makes 1:3 grid of results
sns.barplot(data = vaders, x='category', y='pos', ax= axs[0])
sns.barplot(data = vaders, x='category', y='neu', ax= axs[1])
sns.barplot(data = vaders, x='category', y='neg', ax= axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# postive have higher +ve barplot and neutral have all the three in kind of neutral state same in the negative bar plot

# Roberta pretrained Model
# >>used in order to pick up on the sarcast statement in a  comment.
# using a model trained for a large corpus of data,
# Transformer model accounts for the words but also the context related to others

# In[24]:


#!pip install torch 
#!pip install tensorflow


# In[25]:


get_ipython().system('pip install transformers')
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[26]:


import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# In[27]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
##from huggingface
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# #VADER RESULTS 

# In[28]:


print(ex)
sia.polarity_scores(ex)


# In[29]:


#ROBERTA Model
encoded_text = tokenizer(ex, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg': scores[0],
    'roberta_neu': scores[1],
    'roberta_pos': scores[2]
}
print(scores_dict)


# In[40]:


#ROBERTA Model
def polarity_scores_roberta(example):
    encoded_text = tokenizer(ex, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


# In[31]:


dic = {}
for i, row in tqdm(reddit_df.iterrows(), total=len(reddit_df)):
    try:
        text = row['clean_comment']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        roberta_result = polarity_scores_roberta(text)

        if vader_result is not None and roberta_result is not None:
            both = {f"vader_{key}": value for key, value in vader_result.items()}
            both.update(roberta_result)
            dic[myid] = both
        else:
            # Handle the case when either result is None
            pass

    except RuntimeError:
        print(f'Broke for id {myid}')
        # Add appropriate code here based on your requirements


# In[51]:


dic = {}
for i, row in tqdm(reddit_df.iterrows(), total=len(reddit_df)):
    text = row['clean_comment']
    myid = row["Id"]
    vader_result = sia.polarity_scores(text)
    vader_result_rename = {}
    for key, value in vader_result.items():
        vader_result_rename[f"vader_{key}"] = value
    roberta_result = polarity_scores_roberta(text)
    both = {**vader_result_rename, **roberta_result}
    dic[myid] = both
    


# In[50]:


both


# In[48]:





# In[52]:


output = ""
for myid, result in dic.items():
    output += f"ID: {myid}\n"
    for key, value in result.items():
        output += f"{key}: {value}\n"
    output += "\n"  # Add a blank line between entries

print(output)


# In[53]:


results_df = pd.DataFrame(dic).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(reddit_df, how='left')
#T flips horizontally


# In[54]:


results_df.head()


# #Compare Scores between models

# In[57]:


results_df.columns


# In[67]:


sns.pairplot(data = results_df,
            vars = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound',
       'roberta_neg', 'roberta_neu', 'roberta_pos'],
             hue = 'category', palette = 'Set2')
            
plt.show()


# ##Review Examples
# negative sentiment 1star review

# In[74]:


print(results_df.columns)


# In[75]:


results_df.query('category == 1').sort_values('roberta_pos', ascending = False)['clean_comment'].values[0]


# In[76]:


results_df.query('category == 1').sort_values('vader_pos', ascending = False)['clean_comment'].values[0]


# ##negative sentiment 5star review

# In[78]:


results_df.query('category == -1').sort_values('roberta_neg', ascending = False)['clean_comment'].values[0]


# In[79]:


results_df.query('category == -1').sort_values('vader_neg', ascending = False)['clean_comment'].values[0]


# Transformers Pipeline ##hugging face 

# In[82]:


from transformers import pipeline
sent_pipeline = pipeline("sentiment-analysis")


# In[83]:


sent_pipeline('I want to go hibernate!')


# In[84]:


sent_pipeline('I dont want to go !')


# In[85]:


sent_pipeline('I hate disgusting and smelly people')


# TWITTER REVIEWS ANALYSIS

# In[86]:


twt_df


# In[98]:


a = twt_df['category'].value_counts().sort_index().plot(kind='bar',
                                                      title='Count of Reviews',
                                                      figsize=(10,5))

a.set_xlabel('Category') 
plt.show()


# In[99]:


# Basic nltk
eg = twt_df['clean_text'][50]
print(eg)


# In[100]:


tokens = nltk.word_tokenize(ex)
tokens[:10]


# In[101]:


#declares parts of speech POS
tagged = nltk.pos_tag(tokens) 
tagged[:10]


# In[102]:


#groups the tokens into chunks of text
entities = nltk.chunk.ne_chunk(tagged[:10])
entities.pprint()


# VADER SENTIMENT SCORING

# In[103]:


sia.polarity_scores(eg)


# In[105]:


twt_df['Id'] = twt_df.reset_index().index + 1


# In[106]:


list(twt_df.columns)


# In[107]:


twt_df['clean_text'] = twt_df['clean_text'].astype(str)


# In[108]:


#runs the polarity score over the complete dataset
dic = {}
for i, row in tqdm(twt_df.iterrows(), total=len(twt_df)):
    text = row['clean_text']
    myid = row['Id']
    dic[myid] = sia.polarity_scores(text)


# In[109]:


pd.DataFrame(dic).T
#stores the polarity scores in a pandas dataframe


# In[110]:


vaders = pd.DataFrame(dic).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(twt_df, how='left')
#T flips horizontally
#merges the polarity scores with the original dataset


# In[111]:


vaders


# In[112]:


sns.barplot(data = vaders, x='category', y='compound')
a.set_title('Compound Score by Twitter Star Reviews', loc='center')
plt.tight_layout()
plt.show()


# In[114]:


fig, axs = plt.subplots(1,3, figsize=(12,3)) 
#makes 1:3 grid of results
sns.barplot(data = vaders, x='category', y='pos', ax= axs[0])
sns.barplot(data = vaders, x='category', y='neu', ax= axs[1])
sns.barplot(data = vaders, x='category', y='neg', ax= axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# In[115]:


print(eg)
sia.polarity_scores(eg)


# ROBERTA PRETRAINED MODEL

# In[116]:


#ROBERTA Model
encoded_text = tokenizer(eg, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg': scores[0],
    'roberta_neu': scores[1],
    'roberta_pos': scores[2]
}
print(scores_dict)


# In[117]:


def polarity_scores_roberta(example):
    encoded_text = tokenizer(eg, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


# In[119]:


dic = {}
for i, row in tqdm(twt_df.iterrows(), total=len(twt_df)):
    try:
        text = row['clean_text']
        myid = row['Id']
        vader_res = sia.polarity_scores(text)
        roberta_res = polarity_scores_roberta(text)

        if vader_res is not None and roberta_res is not None:
            both = {f"vader_{key}": value for key, value in vader_res.items()}
            both.update(roberta_res)
            dic[myid] = both
        else:
            # Handle the case when either result is None
            pass

    except RuntimeError:
        print(f'Broke for id {myid}')
        # Add appropriate code here based on your requirements


# In[122]:


dic = {}
for i, row in tqdm(twt_df.iterrows(), total=len(twt_df)):
    text = row['clean_text']
    myid = row["Id"]
    vader_res = sia.polarity_scores(text)
    vader_res_rename = {}
    for key, value in vader_res.items():
        vader_res_rename[f"vader_{key}"] = value
    roberta_res = polarity_scores_roberta(text)
    both = {**vader_res_rename, **roberta_res}
    dic[myid] = both
    


# In[123]:


both


# In[124]:


output = ""
for myid, result in dic.items():
    output += f"ID: {myid}\n"
    for key, value in result.items():
        output += f"{key}: {value}\n"
    output += "\n"  # Add a blank line between entries

print(output)


# In[125]:


results_df = pd.DataFrame(dic).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(twt_df, how='left')
#T flips horizontally


# In[126]:


results_df.head()


# Compare Scores between models

# In[127]:


results_df.columns


# In[128]:


sns.pairplot(data = results_df,
            vars = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound',
       'roberta_neg', 'roberta_neu', 'roberta_pos'],
             hue = 'category', palette = 'Set2')
            
plt.show()


# In[129]:


print(results_df.columns)


# In[130]:


results_df.query('category == 1').sort_values('roberta_pos', ascending = False)['clean_text'].values[0]


# In[131]:


results_df.query('category == 1').sort_values('vader_pos', ascending = False)['clean_text'].values[0]


# ##negative sentiment 5star review

# In[132]:


results_df.query('category == -1').sort_values('roberta_neg', ascending = False)['clean_text'].values[0]


# In[133]:


results_df.query('category == -1').sort_values('vader_neg', ascending = False)['clean_text'].values[0]


# Transformers Pipeline ##hugging face 

# In[138]:


sentiments = []
for text in twt_df['clean_text']:
    result = sent_pipeline(text)
    sentiment = result[0]['label']
    score = result[0]['score']
    sentiments.append((sentiment, score))

twt_df['sentiment'] = sentiments


# In[139]:


for sentiment, score in twt_df['sentiment']:
    print(f"Sentiment: {sentiment}, Score: {score}")


# In[ ]:




