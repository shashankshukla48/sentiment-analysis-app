import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk import pad_sequence
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.python.ops.init_ops import truncated_normal_initializer

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

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
VOCAB_SIZE = 10000
tokenizer  = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)
print("learned vocab of ",len(tokenizer.word_index))
x_train_sequence = tokenizer.texts_to_sequences(x_train)
x_test_sequence = tokenizer.texts_to_sequences(x_test)
from tensorflow.keras.preprocessing.sequence import pad_sequences
MAX_LEN = 200
x_train_padded = pad_sequences(x_train_sequence, maxlen = MAX_LEN,padding='post', truncating='post')
x_test_padded = pad_sequences(x_test_sequence, maxlen = MAX_LEN,padding='post', truncating='post')
lable_mapping = {'positive':1 , 'negative':0}
y_train_final = y_train.map(lable_mapping)
y_test_final = y_test.map(lable_mapping)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
VOCAB_SIZE = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
model = Sequential()
#add ambedding layer
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=200))
#built model mannually
model.build(input_shape=(None, MAX_LEN))
model.add(Dense(units=1, activation='sigmoid'))
print('dense added')
model.compile(
    #add optimizer adam to increase learning rate
    optimizer='adam',
    #add loss binary crossentropy which is for classification probles which has sigmoid activation
    loss='binary_crossentropy',
    metrics=['accuracy']
)
EPOCHS = 5#number of time model will be train on provided dataset
BATCH_SIZE = 64 #batches in which model will be trained
history = model.fit(x_train_padded, y_train_final, epochs=EPOCHS,batch_size=BATCH_SIZE, validation_data=(x_test_padded,y_test_final))

