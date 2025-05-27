import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split        # To split up dataset for model
from sklearn.feature_extraction.text import TfidfVectorizer  # To give natural language numerical values 
from sklearn.svm import LinearSVC                           # Linear Support Vector Classifier

data = pd.read_csv(r'C:\Users\mattg\OneDrive\Desktop\Data_Science_Projects\FakeNewsDetection\fake_or_real_news.csv')

data['fake'] = data['label'].apply(lambda x: 0 if x == 'REAL' else 1)   # Creates column for fake : If label is real, then 1, else 0
data = data.drop('label', axis = 1)

X, y = data['text'], data['fake']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)   # Split up X and y

vectorizor = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
X_train_vectorized = vectorizor.fit_transform(X_train)
X_test_vectorized = vectorizor.transform(X_test)

clf = LinearSVC()           # Initialize LinearSVC model
clf.fit(X_train_vectorized ,y_train)    # Train model using split data

clf.score(X_test_vectorized, y_test)    # Test model

# Print and read onto text.txt and vectorize the text to feed into model
# Can scan any article -> tweak text.txt
with open('text.txt', 'w', encoding = 'utf-8') as f:
    f.write(X_test.iloc[10])

with open('text.txt','r',encoding = 'utf-8') as f:
    text = f.read()

vectorized_text = vectorizor.transform([text])

print(clf.predict(vectorized_text))     # Predict whether or not  1 = Real, 0 = Fake

print(y_test.iloc[10])  # Check if Prediction is Accurate