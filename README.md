# Product Review Sentiment Analysis

This project automates sentiment analysis of product reviews, classifying them into positive or negative sentiments. Leveraging NLP techniques and ML algorithms, it provides businesses with insights into customer feedback.

## Overview

Understanding customer sentiments is crucial for businesses to improve products and services. This project streamlines the sentiment analysis process, allowing businesses to efficiently analyze large volumes of product reviews.

## Key Features

- **Data Preprocessing**: Cleansing, tokenization, and removal of stopwords ensure quality input for analysis.
- **Feature Extraction**: TF-IDF vectorization captures word importance and enables effective model training.
- **Model Training**: Decision Tree Classifier learns patterns in the text data for sentiment prediction.
- **Evaluation**: Accuracy score and confusion matrix visualization assess model performance.
- **Prediction**: A function predicts sentiment for new reviews, facilitating real-time analysis.

## Usage

1. **Data Preparation**: Ensure your dataset (`product_reviews.csv`) contains reviews and corresponding ratings.
2. **Dependencies**: Install necessary dependencies using `pip install -r requirements.txt`.
3. **Training**: Run `train_model.py` to preprocess data, train the model, and evaluate its performance.
4. **Prediction**: Use the `predict_review(review)` function in `predict.py` to predict sentiment for new reviews.

## How to Run the Project

To run the project, follow these steps:

1. Ensure you have Python installed on your system.
2. Clone this repository to your local machine.
3. Navigate to the project directory.
4. Install the required dependencies using `pip install -r requirements.txt`.
5. Prepare your dataset (`product_reviews.csv`) with reviews and corresponding ratings.
6. Run `train_model.py` to preprocess data, train the model, and evaluate its performance.
7. Use the `predict_review(review)` function in `predict.py` to predict sentiment for new reviews.

## Example

```python
from predict import predict_review

new_review = "This product exceeded my expectations! Highly recommended."
print(f"The sentiment is: {predict_review(new_review)}")
