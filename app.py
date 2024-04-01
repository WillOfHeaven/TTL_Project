import streamlit as st
import numpy as np
import pandas as pd
import nltk
import re
import torch
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf
from sklearn.metrics import accuracy_score
@st.cache_data()
def load_model_transformer():
    #loading the transformer model imported from hugging face
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer1 = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model1 = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    return tokenizer1, model1

@st.cache_data()
def load_data_init():
    df=pd.read_csv('amazon.csv')
    df.drop('Unnamed: 0',inplace=True,axis=1) #eliminating the column
    df.isnull().any(axis=0)
    df.dropna(inplace=True)   
    return df


@st.cache_data()
def load_data():
    df1 = model_implementation_nltk()
    #df2 = model_implementation_transformer()
    #df3 = model_implementation_LSTM()
    #return df1,df2,df3
    df2 = model_implementation_transformer()
    return df1,df2

def data_analysis():
    # Load the data
    df = load_data_init()

    # Show the shape of the dataset
    st.subheader("Shape of Dataset")
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])

    # Show the datatypes of thwe features
    st.subheader("Data Types of Features")
    df1 = pd.DataFrame(df.dtypes)
    #df1.drop(df.columns[0], axis=1, inplace=True)
    st.write(df1.T)

    # Show the first few rows of the dataset
    st.subheader("Preview of Data")
    st.write(df.head())

    # Show the summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
   
    # Show the distribution of ratings
    st.subheader("Distribution of overall ratings")
    rating_counts = df['overall'].value_counts()
    #showing the distribution of ratings in a bar chart
    Reviews= ["Positive_reviews", "Negative_reviews", "Neutral_reviews"]
    #Colors for each bar
    No_values= [3997, 644, 274]
    colors= ['lightgreen', 'orange', 'purple']
    #Create bar chart
    fig1, ax1 = plt.subplots()
    ax1.bar(range(len(Reviews)), No_values, color=colors, tick_label=Reviews)
    #Attach text labels above each bar 
    for i in range(len(Reviews)):
         ax1.text(i, No_values[i]+0.2, str(No_values[i]), ha='center')
    #Add Labels and title
    ax1.set_xlabel('Reviews')
    ax1.set_ylabel('No_values')
    ax1.set_title('Response of Customer')
    plt.xticks(rotation=45)  # Rotate labels

    #Show plot
    st.pyplot(fig1)

    
    st.subheader("Pie Chart of overall ratings in percentage")
    # normalize the value counts to show the distribution in percentage
    rating_counts_normalized = df['overall'].value_counts(normalize=True)
    Ratings= ["5-star", "4-star", "1-star", "3-star", "2-star"]
    percentages= rating_counts_normalized.round(2)
    explode=[ 0 if rating!='4-star' else 0.3 for rating in Ratings]
    # Custom colors for each wedge
    colors = ['green', 'red', 'orange', 'purple', 'yellow']
    #    start angle for first wedge
    startangle=90
    #   Add shadow
    shadow=True
    #   Create Pie Charts
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(percentages,explode=explode,labels=Ratings,colors=colors,shadow=shadow)
    #   add legend with header
    ax.legend(title="Overall Ratings", bbox_to_anchor=(1, 0.5), loc="center right")  # Adjust legend location
    ax.legend(title="Overall Ratings", loc="best",bbox_to_anchor=(1,1))
    #   show the plot
    ax.axis("equal")
    st.pyplot(fig)

@st.cache_data()
def model_implementation_nltk():
    df = load_data_init()
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    # Load the data
    rt=lambda x: re.sub("[^a-zA-Z]",' ',str(x))
    df['reviewText']=df['reviewText'].map(rt)
    df['reviewText']=df['reviewText'].str.lower()
    df[['polarity','subjectivity']]=df['reviewText'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
    for index,row in df['reviewText'].items():
        score = SentimentIntensityAnalyzer().polarity_scores(row)
        neg=score['neg']
        neu=score['neu']
        pos=score['pos']
        if neg>pos:
            df.loc[index,'sentiment']="Negative"
        elif pos>neg:
            df.loc[index,'sentiment']="Positive"
        else:
            df.loc[index,'sentiment']="neutral" 
    return df

def sentiment_score(review):
    tokenizer, model = load_model_transformer()
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))
   
@st.cache_data()
def model_implementation_transformer():
    df = load_data_init()
    newdf=df.iloc[:100,1:3]
    newdf['sentiment']=newdf['reviewText'].apply(lambda x: sentiment_score(x[:512]))
    return newdf

def model_comparison():
    df1,df2 = load_data()
    st.header("Model Comparison")
    st.subheader("Sentiment Analysis using nltk.sentiment.vader")
    st.write("Details :")
        # Create two columns
    col1, col2 = st.columns(2)
    # Display value counts in the first column
    col1.subheader("Overall Rating Counts")
    col1.write(df1['overall'].value_counts())
    # Display value counts in the second column
    col2.subheader("Sentiment Value Counts")
    col2.write(df1['sentiment'].value_counts())
    #st.subheader("Accuracy of the Model")
    #df[df["sentiment"]=='Positive'].sort_values("wilson_lower_bound",ascending=False).head(5)
    # Assuming df['overall'] contains the actual values and df['sentiment'] contains the predicted values
    #accuracy = accuracy_score(df['overall'], df['sentiment'])
    # Display the accuracy
    #st.write(f"Accuracy: {accuracy * 100:.2f}%")
    #f['sentiment'].value_counts()
    st.subheader("Sentiment Analysis using transformer model")
    st.write("Details :")
    # Create two columns
    col1, col2 = st.columns(2)
    col1.subheader("Calcualted Sentiment")
    col1.write(df2['sentiment'].value_counts())
    col2.subheader("Actual Rating Counts")
    col2.write(df2['overall'].value_counts())

    st.subheader("Accuracy of the Model")
    st.write("Assuming the overall feature contains the actual values comparing with calculated sentiment containing the predicted values")
    accuracy = accuracy_score(df2['overall'], df2['sentiment'])
    # Display the accuracy
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    sqmean=np.power(df2['overall']-df2['sentiment'],2)
    mse=np.mean(sqmean)
    st.write("Mean Square Error: ",mse)
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    fig, axs = plt.subplots(1, 2, figsize=(10, 7))
    X1 = df2['sentiment'].value_counts()
    X2 = df2['overall'].value_counts()
    labels = ['Excellent', 'Good', 'Neutral', 'Bad', 'Worse']
    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
    
    axs[0].pie(X1, labels=labels, colors=colors, autopct='%1.1f%%')
    axs[0].set_title('Calculated Ratings')

    axs[1].pie(X2, labels=labels, colors=colors, autopct='%1.1f%%')
    axs[1].set_title('Actual Ratings')

    plt.legend(title="Calculated Ratings", bbox_to_anchor=(1,1))
    plt.tight_layout()

    st.pyplot(fig)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Model Comparison"])

    if page == "Home":
        st.title("Tools and Technology Lab")
        st.subheader("Semester Project on Amazon Reviews Analysis")
        
        # Project description
        st.markdown("""
        This project aims to analyze Amazon reviews using various data analysis and machine learning techniques. 
        The goal is to gain insights into customer sentiment and product quality, and to build predictive models 
        for future reviews.
        """)

        # Team members
        st.subheader("Team Members")
        team_data = {
            "Name": ["Rishabh Raj Srivastava","Mohak Pathak","Sambuddha Chatterjee","Kritika Arora", "Shivani Basa"],
            "Role": ["Data Analyst","Data Analyst","Machine Learning Engineer", "Machine Learning Engineer", "Machine Learning Engineer"],
            "Roll no.": ["2105056", "2105286", "2105485", "2105491", "2105551"]
        }
        team_df = pd.DataFrame(team_data)
        st.table(team_df)

    elif page == "Data Analysis":
        st.header("Data Analysis")
        data_analysis()
        # Add content related to Model 1 here
    elif page == "Model Comparison":
        st.header("Model Comparison")
        model_comparison()
        # Add content related to Model 2 here
    
if __name__ == "__main__":
    main()