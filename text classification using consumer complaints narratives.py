#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[4]:


df = pd.read_csv('complaints.csv')


# In[5]:


print(df.head())


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Product', order=df['Product'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Product')
plt.ylabel('Count')
plt.title('Count of Complaints by Product')
plt.show()


# In[10]:


from wordcloud import WordCloud

complaint_text = ' '.join(df['Consumer complaint narrative'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(complaint_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Consumer Complaint Narratives')
plt.show()


# In[11]:


'''def text_preprocessing(text):
    # Implement your text pre-processing steps here
    # Example: Convert to lowercase and remove punctuation
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

df['Consumer complaint narrative'] = df['Consumer complaint narrative'].apply(text_preprocessing)'''

import pandas as pd
import numpy as np  # Import numpy for handling NaN values

def text_preprocessing(text):
    # Check if the input is a string (not NaN)
    if isinstance(text, str):
        # Implement your text pre-processing steps here
        # Example: Convert to lowercase and remove punctuation
        text = text.lower()
        text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

# Apply the text_preprocessing function to the 'Consumer complaint narrative' column
df['Consumer complaint narrative'] = df['Consumer complaint narrative'].apply(text_preprocessing)

# Replace NaN values with an empty string or any other suitable value
df['Consumer complaint narrative'] = df['Consumer complaint narrative'].replace(np.nan, '')

# Now, the function should handle NaN values without errors


# In[12]:


# Step 3: Split data into train and test sets
X = df['Consumer complaint narrative']  # Features (text data)
y = df['Product']  # Target variable (product category)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


# Step 4: Text Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)



# In[14]:


# Step 5: Train a Multi-Class Classification Model (Random Forest in this example)
#clf = RandomForestClassifier(n_estimators=100, random_state=42)
#clf.fit(X_train_tfidf, y_train)
'''from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)'''

from sklearn.naive_bayes import MultinomialNB

# Initialize the Multinomial Naive Bayes classifier
clf = MultinomialNB()

# Define the number of iterations or batches
n_iterations = 100  # Adjust as needed

for i in range(n_iterations):
    # Assuming X_train_tfidf and y_train are your training data
    clf.partial_fit(X_train_tfidf, y_train, classes=np.unique(y_train))

    # Calculate the training progress as a percentage
    progress = (i + 1) / n_iterations * 100
    print(f"Training progress: {progress:.2f}%")

# Training is complete
print("Training complete!")


# In[15]:


# Step 6: Model Evaluation
y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')



# In[16]:


# Print classification report for detailed metrics
print(classification_report(y_test, y_pred))


# Classification algorithms:Multinomial Naive Bayes
