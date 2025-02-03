import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.markdown("""
<div style='background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
<h3 style='color: #2c3e50; margin-bottom: 10px;'>About this Model</h3>
<p style='color: #34495e;'>
This model uses GRU (Gated Recurrent Unit) neural network architecture to predict the next word in Nepali sentences. 
The model was trained on a dataset of Nepali text scraped from Kaggle. It learns patterns in Nepali language to make predictions.
</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    model = load_model("next_word_lstm.h5")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()
max_sequence_len = model.input_shape[1] + 1

# Initialize session states
if "text" not in st.session_state:
    st.session_state.text = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# Custom CSS for better UI
st.markdown("""
<style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f4f4f9;
        color: #333;
    }
    
    .prediction-box {
        background: #e8f7ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }

    .nepali-text {
        font-size: 1.2em;
        color: #2c3e50;
    }

    .instruction-box {
        background: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-weight: bold;
        width: 75%;
    }
    .text-box {
        background: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-weight: bold;
    }

    .prediction-label {
        color: #4CAF50;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .button-section {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }

    .button-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 15px 0;
        padding: 10px;
    }

    .button-primary {
        background-color: #4CAF50;
        color: white;
        padding: 8px 16px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }

    .button-secondary {
        background-color: #2196F3;
        color: white;
        padding: 8px 16px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }

   
    .button-primary:hover, .button-secondary:hover {
        opacity: 0.9;
    }

    .prediction-box:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* Styling for the text area */
    .stTextArea textarea {
        background-color: #809080;
        color: #000000;
        border: 2px solid #000000;
        border-radius: 8px;
        padding: 15px;
        font-size: 16px;
        font-family: 'Arial', sans-serif;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    .stTextArea textarea:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
        outline: none;
    }

    .stTextArea textarea::placeholder {
        color: #000000;
        font-style: italic;
    }

</style>
""", unsafe_allow_html=True)

st.title("Nepali Next Word Predictor")

# Instructions
st.markdown("""
<div class='instruction-box'>
    <b>Instructions:</b><br>
    - This model is for Nepali language only.<br>
    - Press Enter to see prediction.<br>
    - Press Clear and type next word to get prediction.
</div>
""", unsafe_allow_html=True)

# Create two columns for main content and examples
main_col, example_col = st.columns([3, 1])

with main_col:
    # Text area with placeholder
    text = st.text_area(
        "",
        value=st.session_state.text,
        height=150,
        placeholder="यहाँ नेपाली मा टाइप गर्नुहोस्... ",
        key="text_input",
        on_change=lambda: setattr(st.session_state, 'prediction', None)
    )

    # Restore previous button layout
    col1, col2= st.columns([2,2])
    with col2:
        enter_pressed = st.button("Enter", use_container_width=True)
    with col1:
        if text and st.button("Clear", use_container_width=True):
            st.session_state.text = ""
            st.session_state.prediction = None
            st.rerun()

    # Update prediction on enter
    if text and enter_pressed:
        st.session_state.text = text
        current_prediction = predict_next_word(model, tokenizer, text, max_sequence_len)
        st.session_state.prediction = current_prediction
        
        if st.session_state.prediction:
            st.markdown(f"""
            <div class='prediction-box'>
                <div class='prediction-label'>सम्भावित अर्को शब्द: </div>
                <div class='nepali-text'>{st.session_state.prediction}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("❗ कुनै भविष्यवाणी उपलब्ध छैन / No prediction available")
st.markdown("</div>", unsafe_allow_html=True)
