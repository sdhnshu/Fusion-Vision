import os
import sys
sys.path.append(".")
import json
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

from text_classification import config
from text_classification import predict
from text_classification import utils


def softmax(x):
    x = x - max(x)
    y = np.exp(x)
    return y / sum(y)


def normalize(x):
    return (x-min(x)) / (max(x)-min(x))


# Title
st.title("Creating an End-to-End ML Application")
st.write("""[<img src="https://github.com/madewithml/images/blob/master/images/yt.png?raw=true" style="width:1.2rem;"> Watch Lesson](https://www.youtube.com/madewithml?sub_confirmation=1) · [<img src="https://github.com/madewithml/images/blob/master/images/github_logo.png?raw=true" style="width:1.1rem;"> GitHub](https://github.com/madewithml/e2e-ml-app-pytorch) · [<img src="https://avatars0.githubusercontent.com/u/60439358?s=200&v=4" style="width:1.2rem;"> Made With ML](https://madewithml.com)""", unsafe_allow_html=True)
st.write("Video lesson coming soon...")

# Get best run
project = 'GokuMohandas/e2e-ml-app-pytorch'
best_run = utils.get_best_run(project=project,
                              metric="test_loss", objective="minimize")

# Load best run (if needed)
best_run_dir = utils.load_run(run=best_run)

# Get run components for prediction
args, model, X_tokenizer, y_tokenizer = predict.get_run_components(
    run_dir=best_run_dir)

# Pages
page = st.sidebar.selectbox(
    "Choose a page", ['Prediction', 'Model details'])
if page == 'Prediction':

    st.header("🚀 Try it out!")

    # Input text
    text = st.text_input(
        "Enter text to classify", value="The Canadian government officials proposed the new federal law.")

    # Predict
    results = predict.predict(inputs=[{'text': text}], args=args, model=model,
                              X_tokenizer=X_tokenizer, y_tokenizer=y_tokenizer)

    # Results
    raw_text = results[0]['raw_input']
    st.write("**Raw text**:", raw_text)
    preprocessed_text = results[0]['preprocessed_input']
    st.write("**Preprocessed text**:", preprocessed_text)
    st.write("**Probabilities**:")
    st.json(results[0]['probabilities'])

    # Interpretability
    st.write("**Top n-grams**:")
    words = preprocessed_text.split(' ')
    top_n_grams = {}
    token_index_to_freq = {i: 0 for i in range(len(words))}
    for filter_size, d in results[0]['top_n_grams'].items():
        top_n_grams[filter_size] = d['n_gram']
        start = int(d['start'])
        end = int(d['end'])
        for i in range(start, end):
            token_index_to_freq[i] += 1
    softmax_values = softmax(np.fromiter(token_index_to_freq.values(), dtype=int))
    normalized_values = normalize(x=softmax_values)

    # Heatmap
    html = "<div>"
    for i, token in enumerate(words):
        if token == '<UNK>': token = 'UNK'
        html += f'<span style="background-color: hsl(100, 100%, {((1-normalized_values[i]) * 50 + 50)}%">' + \
            token + ' </span>'
    html += "</div><br>"
    st.write(html, unsafe_allow_html=True)
    st.json(top_n_grams)

    # Warning
    st.info("""⚠️The model architecture used in this demo is **not state-of-the-art** and it's not **fully optimized**, as this was not the main focus of the lesson.
            Also keep in mind that the **dataset** is dated and **limited** to particular vocabulary.
            If you are interested in generalized text classification or NLP in general, check out these [curated resources](https://madewithml.com/topics/#nlp).""")

    # Show raw json
    show_json = st.checkbox("Show complete JSON output:", value=False)
    if show_json:
        st.json(results)


elif page == 'Model details':

    st.header("All Experiments")
    st.write(f'[https://app.wandb.ai/{project}](https://app.wandb.ai/{project})')

    st.header("Best Run")

    # Run details
    st.write(f"**Name**: {best_run._attrs['displayName']} ({best_run._attrs['name']})")
    st.write("**Timestamp**:", best_run._attrs['createdAt'])
    st.write(f"**Runtime**: {best_run._attrs['summaryMetrics']['_runtime']:.1f} seconds")

    # Performance
    st.write("**Performance**:")
    performance = utils.load_json(
        os.path.join(best_run_dir, 'performance.json'))
    st.json(performance)

    # Confusion matrix
    st.image(os.path.join(best_run_dir, 'confusion_matrix.png'))

    # Config
    st.write("**Config**:")
    st.json(best_run._attrs['config'])
