import pandas as pd 
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score ,confusion_matrix

nltk.download('stopwords')
nltk.download('punkt')
#IMPORTING AND LABELLING THE DATA   
data = pd.read_csv('data.csv' )
test = data.head(2)
# data.dropna(subset=data['rating'])

data['label'] = np.where(data['rating'] < 3, 0 , 1 )


# test = data.head(2)
# print(test)

#PREPROCSSING THE TEXT
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

data['processed_review'] = data['review'].apply(preprocess_text)

text = ' '.join(data['processed_review'])

# Create a word cloud object
# font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'  # Update with the correct path on your system
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# # Display the word cloud using matplotlib
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')  # Turn off the axis
# plt.show()

cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(data['processed_review']).toarray()
# print(X)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.33,stratify=data['label'], random_state = 42)

model = DecisionTreeClassifier(random_state=0) 
model.fit(X_train,y_train) 
  
#testing the model 
pred = model.predict(X_train) 
print(accuracy_score(y_train,pred))



         
from sklearn import metrics 
cm = confusion_matrix(y_train,pred) 
  
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,  display_labels = [False, True]) 
  
# cm_display.plot() 
# plt.show()


def predict_review(review):
    # Preprocess the review
    processed_review = preprocess_text(review)
    # Vectorize the review
    review_vector = cv.transform([processed_review]).toarray()
    # Predict the label
    prediction = model.predict(review_vector)
    # Convert prediction to readable label
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Example usage
new_review = "This product is bad, I hate it!"
print(f"The review is: {predict_review(new_review)}")