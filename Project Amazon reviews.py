#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install cufflinks')
get_ipython().system('pip install plotly')


# In[8]:


get_ipython().system('pip install textblob')


# In[111]:


get_ipython().system('pip install vaderSentiment')


# In[10]:


get_ipython().system('pip install wordcloud')


# In[92]:


import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
cf.go_offline();
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")


# In[93]:


df=pd.read_csv('amazon.csv')
df.head(20)


# # Total number of rows and columns present in the table

# In[94]:


df.shape


# # Total number of entries in the table

# In[95]:


df.size


# # Columns present in the table

# In[96]:


df.columns.tolist()


# # Eliminating the column "Unnamed: 0"

# In[97]:


df.drop('Unnamed: 0',inplace=True,axis=1) #eliminating the column
df.head()


# # No of rows and columns after eliminating the column

# In[98]:


df.shape


# # Datatypes of all the column

# In[99]:


df.dtypes


# # Checking if there is any duplicated Value

# In[100]:


df[df.duplicated()]


# thus we get to know that there are no duplicate values present in the table

# # Checking if there are any null values or not

# In[101]:


df[df.isnull().any(axis=1)]


# # The total number of reviewers

# In[102]:


df["overall"].value_counts()


# # To represent the values in the bar graph

# In[103]:


Ratings= ["5-star", "4-star", "1-star", "3-star", "2-star"]
No_values= [3922, 527, 244, 142, 80]
#Colors for each bar
colors= ['lightgreen', 'orange', 'purple', 'red', 'yellow']
#Create bar chart
plt.bar(Ratings, No_values, color=colors)

#Attach text labels above each bar 
for i in range(len(Ratings)):
     plt.text(i, No_values[i]+0.2, str(No_values[i]), ha='center')
#Add Labels and title
plt.xlabel('Ratings')
plt.ylabel('No_values (%)')
plt.title('Overall Reviews of Customer')

#Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[104]:


df["overall"].value_counts(normalize=True)


# # To represent the Percentage in ratings

# In[105]:


Ratings= ["5-star", "4-star", "1-star", "3-star", "2-star"]
Percentages= [0.79, 0.10, 0.04, 0.02, 0.01]
explode=[ 0 if rating!='4-star' else 0.3 for rating in Ratings]

# Custom colors for each wedge
colors = ['green', 'red', 'orange', 'purple', 'yellow']

#start angle for first wedge
startangle=90

#Add shadow
shadow=True

#Create Pie Charts
plt.pie(Percentages,explode=explode,labels=Ratings,colors=colors,shadow=shadow)

#add legend with header
plt.legend(title="Overall Ratings", loc="best")

#show the plot
plt.axis("equal")
plt.show


# # Comments given by the user on the product

# In[127]:


df['reviewText'].head()


# In[128]:


review_example= df.reviewText[2031]
print(review_example)


# In[129]:


review_example= re.sub("[^a-zA-Z]",' ',review_example)
print(review_example)


# In[130]:


#rt=lambda x: re.sub("[^a-zA-Z]",' ',str(x))
#f['reviewText']=df['reviewText'].map(rt)
review_example=review_example.lower().split()
review_example


# In[143]:


rt=lambda x: re.sub("[^a-zA-Z]",' ',str(x))
df['reviewText']=df['reviewText'].map(rt)
df['reviewText']=df['reviewText'].str.lower()
df.tail(5)


# # Dividing all the sentiments on the basis of Positive, Negative and Neutral

# In[132]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
df[['polarity','subjectivity']]=df['reviewText'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index,row in df['reviewText'].items():
    score=SentimentIntensityAnalyzer().polarity_scores(row)
    neg=score['neg']
    neu=score['neu']
    pos=score['pos']
    if neg>pos:
        df.loc[index,'sentiment']="Negative"
    elif pos>neg:
        df.loc[index,'sentiment']="Positive"
    else:
        df.loc[index,'sentiment']="neutral"


# In[133]:


df[df["sentiment"]=='Positive'].sort_values("wilson_lower_bound",ascending=False).head(5)


# # Counting the Sentiment Values

# In[144]:


df['sentiment'].value_counts()


# # Ploting it into a graph

# In[138]:


Reviews= ["Positive_reviews", "Negative_reviews", "Neutral_reviews"]
No_values= [3997, 644, 274]
#Colors for each bar
colors= ['lightgreen', 'orange', 'purple']
#Create bar chart
plt.bar(Reviews, No_values, color=colors)

#Attach text labels above each bar 
for i in range(len(Reviews)):
     plt.text(i, No_values[i]+0.2, str(No_values[i]), ha='center')
#Add Labels and title
plt.xlabel('Reviews')
plt.ylabel('No_values')
plt.title('Response of Customer')

#Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # Finding the percentiles of the reviews

# In[140]:


df['sentiment'].value_counts(normalize=True)


# # Ploting the percentiles of reviews in a Pie-Chart

# In[141]:


Reviews= ["Positive_reviews", "Negative_reviews", "Neutral_reviews"]
Percentages= [0.81, 0.13, 0.05]
explode=[ 0 if review!='Negative_reviews' else 0.3 for review in Reviews]

# Custom colors for each wedge
colors = ['green', 'red', 'orange']

#start angle for first wedge
startangle=90

#Add shadow
shadow=True

#Create Pie Charts
plt.pie(Percentages,explode=explode,labels=Reviews,colors=colors,shadow=shadow)

#add legend with header
plt.legend(title="Reviews", loc="best")

#show the plot
plt.axis("equal")
plt.show

