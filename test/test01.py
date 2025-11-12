import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

file_path = r'C:\Users\Shashank Shukla\sentiment_analysis\data\IMDB Dataset.csv\IMDB Dataset.csv'
df = pd.read_csv(file_path)
print("data frame loaded")
#lowercase data
def to_lowercase(text):
    return text.lower()
#remove html tags
def remove_html_tags(text):
    pattern = re.compile('<.*?.>')
    return pattern.sub('', text)
#remove puntuations
def remove_puntuation(text):
    return re.sub(r'[^a-z0-9\s]', ' ', text).strip()# This function uses a regular expression to find and remove any character that is not a lowercase letter (a-z), a digit (0-9), or a whitespace character.
#create toen and remove stopwords
english_stopword_list = stopwords.words('english')
stop_words_set = set(english_stopword_list)
def tokenization_and_stopwords_remove(text):
    token = text.split()
    cleaned_token = [token for token in token if token not in stop_words_set]
    return cleaned_token
#lemmatize
lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(tokens):
    lemmatize_tokens = [lemmatizer.lemmatize(tokens) for tokens in tokens]
    return lemmatize_tokens
#rejoin string
def join_tokens(tokens):
    return ' '.join(tokens)
df['cleaned_review'] = df['review'].apply(to_lowercase).apply(remove_html_tags).apply(tokenization_and_stopwords_remove).apply(lemmatize_tokens).apply(join_tokens)
print(df[['review','cleaned_review','sentiment']].head())


#x will be the cleaned data and y will be the sentiment to train data
x = df['cleaned_review']
y = df['sentiment']
# print('------Features(x)--------')
# print(x.head())
# print("\n" + "="*50 + "\n")
# print("--- Target (y) ---")
# print(y.head())

#Import the train_test_split Function
#split data into training and testing set
#Scikit-learn's Dedicated Tool for the Job
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42,stratify=y)

tfidf_vectorizer = TfidfVectorizer(max_features=10000)
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)

#model training
lr_model = LogisticRegression(random_state=42)
lr_model.fit(x_train_tfidf,y_train)#training model
y_pred = lr_model.predict(x_test_tfidf)
#train model to more precise using confuddion matrix
#classification_report this function generates text based report that  shows precision recall f1 score
#confusion matrix this function directly com[utes raw numbers for confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
report = classification_report(y_test,y_pred)

