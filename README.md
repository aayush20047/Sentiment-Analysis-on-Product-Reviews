Project Overview: Sentiment Analysis on Product Reviews

This project aims to build a sentiment analysis model that can classify product reviews as either positive or negative. The analysis uses Natural Language Processing (NLP) techniques to preprocess the text data, transform it into numerical features, and apply a machine learning algorithm for classification. The ultimate goal is to provide an automated way to assess customer feedback, which can be beneficial for businesses to understand their customers better.
Steps Involved

    Data Import and Labeling:
        The dataset (product_reviews.csv) is loaded into a Pandas DataFrame.
        Reviews with ratings less than 3 are labeled as negative (0), and those with ratings 3 or higher are labeled as positive (1).

    Text Preprocessing:
        The preprocess_text function cleans the reviews by removing special characters, extra spaces, and converting text to lowercase.
        Stop words are removed to reduce noise in the data.
        Tokenization is performed to break down sentences into individual words.

    Feature Extraction:
        TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer is used to transform the processed text data into numerical features suitable for machine learning algorithms.
        A maximum of 2500 features are extracted to represent the text data efficiently.

    Model Training and Evaluation:
        The dataset is split into training and test sets with a 67%-33% ratio, ensuring the class distribution is maintained using stratification.
        A Decision Tree Classifier is trained on the training set.
        The model's performance is evaluated using the accuracy score on the training set, and the confusion matrix is displayed for better understanding of the classification results.

    Prediction on New Reviews:
        A function predict_review is created to preprocess, vectorize, and predict the sentiment of new reviews using the trained model.
        Example: The review "This product is bad, I hate it!" is predicted as negative.

Visualization

    A word cloud can be generated to visualize the most frequent words in the reviews, providing insights into common themes and topics.
