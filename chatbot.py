# Save the Streamlit code in a Python script
with open("app.py", "w") as f:
    f.write("""
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the FLAN-T5 model and tokenizer
model_name = "google/flan-t5-large"  # You can use flan-t5-xxl for better quality
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Streamlit app configuration
st.title("AI Learning Assistant")
st.write("Ask me anything about Artificial Intelligence (AI), e.g., 'What is AI?', 'Explain deep learning', or 'What is reinforcement learning?'")

# User input
user_input = st.text_input("Your question:", "")

# Generate response
if user_input:
    inputs = tokenizer(user_input, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=300, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write("### Response:")
    st.write(response)

st.write("Type your question above and hit Enter to get a response. Type 'exit' to quit.")
    """)

