import pickle
import re
import nltk 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

##download the necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

## load the vectorizer
tf = pickle.load(open("artifacts/tf.pkl", 'rb'))

## load the model
def text_preprocessing(text):
  ## lower case
  text = text.lower()
  ## special charcter
  text = re.sub('[^a-zA-z]', ' ', text)
  ## Tokinzation
  text = word_tokenize(text)
  ## stopwords
  text = [word for word in text if word not in stop_words]
  ## lemmetization
  text = [lemmatizer.lemmatize(word) for word in text]
  text = ' '.join(text)
  text = tf.transform([text])
  return text

