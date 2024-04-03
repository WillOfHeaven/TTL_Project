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
import json
import keras
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
    newdf=df.iloc[:1000,1:3]
    newdf['sentiment']=newdf['reviewText'].apply(lambda x: sentiment_score(x[:512]))
    return newdf

@st.cache_data()
def model_implementation_LSTM():
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.model_selection import train_test_split
    df1 = load_data_init()
    max_features = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie revie
    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    from tensorflow.keras.preprocessing.text import Tokenizer
    # Initialize a tokenizer
    tokenizer = Tokenizer(num_words=max_features)
    # Fit the tokenizer on your text data
    X = df1['reviewText']
    tokenizer.fit_on_texts(X)
    # Convert the text data into sequences of integers
    X_sequences = tokenizer.texts_to_sequences(X)
    # Split the data into training and test sets
    df1['custom_overall'] = df1['overall'].apply(lambda x: 1 if x > 2 else 0)
    Y = df1['custom_overall']
    x_train, x_test, y_train, y_test = train_test_split(X_sequences, Y, test_size=0.2, random_state=42)
    # Pad the sequences
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
    model.save('my_model.h5')  # Creates a HDF5 file 'my_model.h5'
    history_dict = history.history
    with open('history.json', 'w') as f:
        json.dump(history_dict, f)
    return history,model

def model_comparison():
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import load_model
    max_features = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    
    tokenizer = Tokenizer(num_words=max_features)
    df1,df2 = load_data()
    st.subheader("Assumptions")
    st.write(" 1) The overall feature contains the actual values of the ratings.")
    st.write(" 2) Positive vs Negative Classification: Classifying reviews with 4-5 stars as positive and 1-3 stars as negative.")    
    st.subheader("Sentiment Analysis using nltk.sentiment.vader")
    st.write("Details :")
        # Create two columns
    df1['custom_overall'] = df1['overall'].apply(lambda x: 'Positive' if x > 3 else 'Negative')
    col1, col2 = st.columns(2)
    # Display value counts in the first column
    col1.subheader("Overall Rating Counts")
    col1.write(df1['custom_overall'].value_counts())
    # Display value counts in the second column
    col2.subheader("Sentiment Value Counts")
    col2.write(df1['sentiment'].value_counts())
    df1['custom_overall'] = df1['overall'].apply(lambda x: 'Positive' if x > 3 else 'Negative')
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
    col2.subheader("Calcualted Sentiment")
    col2.write(df2['sentiment'].value_counts())
    col1.subheader("Actual Rating Counts")
    col1.write(df2['overall'].value_counts())

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
    st.subheader("Sentiment Analysis using LSTM")
    st.write("Details :")
    model = load_model('my_model.h5')
    with open('history.json', 'r') as f:
        loaded_history = json.load(f)
    history = loaded_history
    # Plot the training and validation accuracy
    st.subheader('Training and Validation Accuracy')
    fig, ax = plt.subplots()
    ax.plot(history['accuracy'], label='Train Accuracy')
    ax.plot(history['val_accuracy'], label='Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    st.pyplot(fig)

    # Plot the training and validation loss
    st.subheader('Training and Validation Loss')
    fig, ax = plt.subplots()
    ax.plot(history['loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)
    df = load_data_init()
    df_n = df.iloc[:,1:3]
    review_sequences = tokenizer.texts_to_sequences(df_n['reviewText'])
    review_padded = pad_sequences(review_sequences, maxlen=maxlen)
    # Predict the sentiment of the reviews
    st.subheader("Accuracy of the Model for the first 100 rows")
    df_n['overall_custom'] = df_n['overall'].apply(lambda x: 1 if x > 3 else 0)
    df_n['calculated'] = model.predict(review_padded)
    df_n['calculated_rating'] = df_n['calculated'].apply(lambda x: 1 if x > 0.5 else 0)
    st.write("Comparison of Actual and Calculated Ratings")
    col1, col2 = st.columns(2)
    col1.subheader("Actual Ratings")
    col1.write(df_n['overall_custom'].value_counts())
    col2.subheader("Calculated Ratings")
    col2.write(df_n['calculated_rating'].value_counts())

    accuracy = accuracy_score(df_n['overall_custom'], df_n['calculated_rating'])
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    st.header("Sample implementation of the model")
    # Get the review from the user
    review = st.text_input("Enter your review", "Type Here")
    if st.button("Submit"):
        # Convert the review to a sequence of integers
        review_sequence = tokenizer.texts_to_sequences([review])
        # Pad the sequence
        review_padded = keras.preprocessing.sequence.pad_sequences(review_sequence, maxlen=maxlen)
        # Predict the sentiment of the review
        sentiment = model.predict(review_padded)
        # Write the sentiment to the screen
        #st.write("Sentiment of the review is: ", sentiment)
        if(sentiment>0.5):
            st.write("The review is Positive")
        else:
            st.write("The review is Negative")
        
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
            "Name": ["Rishabh Raj Srivastava","Mohak Pathak","Sambuddha Chatterjee","Kritica Arora", "Shivani Basa"],
            "Role": ["Data Analyst","Data Analyst","Machine Learning Engineer", "Machine Learning Engineer", "Machine Learning Engineer"],
            "Roll no.": ["2105056", "2105286", "2105485", "2105551","2105491"]
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
