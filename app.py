import helper
import pickle
import streamlit as st

# Load the model
model = pickle.load(open('artifacts/lr.pkl', 'rb'))

# Title
st.title('Sentiment Analysis :blue[ _Reviews APP_ ]')
text = st.text_input('Enter your review here')




text = helper.text_preprocessing(text)
if st.button('Predict'):
  prediction = model.predict(text)
  if prediction:
    st.write('Positive Review')
  else:
    st.write('Negative Review')