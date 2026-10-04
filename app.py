import streamlit as st
import joblib
import os

# Configure the Streamlit page
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #ced4da;
    }
    .stButton button {
        border-radius: 20px;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Spam Message Detection")
st.markdown("Enter a message below to check whether it's **Spam** or **Not Spam**.")

model_path = "spam_model.pkl"

if not os.path.exists(model_path):
    st.error("Model not found. Please run `python train.py` to train and generate the model first.")
else:
    # Load the trained model
    @st.cache_resource
    def load_model():
        return joblib.load(model_path)
        
    model = load_model()
    
    # Input area
    message = st.text_area("Message to analyze:", height=150, placeholder="Type your message here...")
    
    # Predict button
    if st.button("Predict"):
        if message.strip() == "":
            st.warning("Please enter a message to predict.")
        else:
            with st.spinner("Analyzing..."):
                prediction = model.predict([message])[0]
                
                # Display result
                st.markdown("### Result:")
                if prediction == 'spam':
                    st.error("🚨 **SPAM DETECTED** 🚨")
                    st.markdown(f"> *{message}*")
                else:
                    st.success("✅ **NOT SPAM (Ham)** ✅")
                    st.markdown(f"> *{message}*")
