import streamlit as st
from emotionClassification.utils.config import Config
from emotionClassification.configs.config import CFGLog
from emotionClassification.executor.inferrer import Inferrer    

config = Config.from_json(CFGLog)
model_name = config.output.model_name

st.title("Emotion Classification")
input_document = st.text_area(label="Enter how you feel")
inferrer = Inferrer()

result = str(inferrer.infer(document=input_document)[0])
st.write(f"Predicted emotion: {result}")
st.write(f"Model used: {model_name}")