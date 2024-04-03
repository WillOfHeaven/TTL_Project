## Amazon Reviews Analysis with Streamlit

This project analyzes Amazon reviews using various data analysis and machine learning techniques. The goal is to gain insights into customer sentiment and product quality, and to build models for predicting future review sentiment.

**Getting Started**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your_username>/amazon-reviews-analysis.git
   ```

2. **Install dependencies:**

   ```bash
   pip install streamlit transformers nltk pandas numpy matplotlib seaborn cufflinks keras tensorflow
   ```

3. **Run the application:**

   ```bash
   python main.py
   ```

   A Streamlit application will open in your web browser at http://localhost:8501.

**Project Structure**

```
amazon-reviews-analysis/
├── data/  # (Optional) Folder to store the Amazon review dataset
├── main.py  # Main script to run the Streamlit application
├── data_analysis.py  # Functions for data loading, analysis, and visualization
├── model_implementation_nltk.py  # Sentiment analysis using NLTK
├── model_implementation_transformer.py  # Sentiment analysis using Transformer model
├── model_implementation_LSTM.py  # Sentiment analysis using LSTM model (pre-trained)
├── sentiment_score.py  # Function to calculate sentiment score using Transformer
├── utils.py  # Helper functions for common tasks
├── requirements.txt  # File containing required dependencies
```

**Features**

* **Data Exploration:** View data shape, data types, summary statistics, and distribution of ratings and sentiment.
* **Sentiment Analysis Comparison:** Compare the performance of NLTK, Transformer, and LSTM models for sentiment analysis. 
* **Interactive Review Analysis:** Enter your own review to see the predicted sentiment.

**Team Members**

* Rishabh Raj Srivastava: [https://github.com/](https://github.com/)<replace_with_your_github_username1> (Data Analyst)
* Mohak Pathak: [https://github.com/](https://github.com/)<replace_with_your_github_username2> (Data Analyst)
* Sambuddha Chatterjee: [https://github.com/](https://github.com/)<replace_with_your_github_username3> (Machine Learning Engineer)
* Kritica Arora: [https://github.com/](https://github.com/)<replace_with_your_github_username4> (Machine Learning Engineer)
* Shivani Basa: [https://github.com/](https://github.com/)<replace_with_your_github_username5> (Machine Learning Engineer)

**Note**

* This repository does not include the Amazon review dataset. You can find publicly available datasets online.
* The LSTM model is assumed to be pre-trained. If you want to train your own LSTM model, you will need to modify the `model_implementation_LSTM.py` script.

We welcome contributions to this project! Feel free to fork the repository and submit pull requests with your improvements.
