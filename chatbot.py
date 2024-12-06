import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load FLAN-T5 model and tokenizer
@st.cache_resource  # Cache the model for faster loading
def load_model():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Initialize model and tokenizer
tokenizer, model = load_model()

# Function to generate AI responses
def get_response(question):
    inputs = tokenizer(question, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=300, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.title("AI Chatbot")
st.write("Ask me anything about Artificial Intelligence!")

# User input
user_input = st.text_input("Type your question here:")
if st.button("Ask") and user_input:
    with st.spinner("Thinking..."):
        answer = get_response(user_input)
    st.success("Here is the answer:")
    st.write(answer)
