import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set Streamlit Page Configuration (MUST BE FIRST)
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

# Load FLAN-T5 model and tokenizer
@st.cache_resource
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

# Streamlit Layout
st.markdown("<div class='header'><h1>🤖 AI Chatbot</h1></div>", unsafe_allow_html=True)
st.write("Welcome! Ask me anything about Artificial Intelligence. Let's learn together! 🚀")

# Chatbot Input
user_input = st.text_input("Type your question below:", placeholder="E.g., What is machine learning?")
if st.button("Ask"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            answer = get_response(user_input)
        st.markdown(
            f"<div class='response-box'><b>Answer:</b> {answer}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Please enter a question!")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ using Streamlit and Hugging Face Transformers</p>", unsafe_allow_html=True)
