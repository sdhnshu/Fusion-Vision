import os
import sys
sys.path.append(".")
from argparse import ArgumentParser
from argparse import Namespace
import collections
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from text_classification import config
from text_classification import data
from text_classification import models
from text_classification import utils


def get_run_components(run_dir):
    # Load args
    config = utils.load_json(
        os.path.join(run_dir, 'config.json'))
    args = Namespace(**config)

    # Load tokenizers
    X_tokenizer = data.Tokenizer.load(
        fp=os.path.join(run_dir, 'X_tokenizer.json'))
    y_tokenizer = data.LabelEncoder.load(
        fp=os.path.join(run_dir, 'y_tokenizer.json'))

    # Load model
    model = models.TextCNN(
        embedding_dim=args.embedding_dim, vocab_size=len(X_tokenizer)+1,
        num_filters=args.num_filters, filter_sizes=args.filter_sizes,
        hidden_dim=args.hidden_dim, dropout_p=args.dropout_p,
        num_classes=len(y_tokenizer.classes))
    model.load_state_dict(torch.load(os.path.join(run_dir, 'model.pt')))
    device = torch.device('cuda' if (
        torch.cuda.is_available() and args.cuda) else 'cpu')
    model = model.to(device)

    return args, model, X_tokenizer, y_tokenizer


def get_probability_distribution(y_prob, classes):
    results = {}
    for i, class_ in enumerate(classes):
        results[class_] = np.float64(y_prob[i])
    sorted_results = {k: v for k, v in sorted(
        results.items(), key=lambda item: item[1], reverse=True)}
    return sorted_results


def get_top_n_grams(tokens, conv_outputs, filter_sizes):
    # Process conv outputs for each unique filter size
    n_grams = {}
    for i, filter_size in enumerate(filter_sizes):

        # Identify most important n-gram for each filter's output
        popular_indices = collections.Counter(
            np.argmax(conv_outputs[filter_size], axis=1))

        # Get corresponding text
        start = popular_indices.most_common(2)[-1][0]
        end = min(len(tokens), start+filter_size)
        n_gram = " ".join([token for token in tokens[start:end]])
        n_grams[filter_size] = {
            'n_gram': n_gram,
            'start': np.float64(start),
            'end': np.float64(end)
            }

    return n_grams


def predict_step(model, dataloader, filter_sizes, device):
    """Predict step."""
    model.eval()
    conv_outputs = collections.defaultdict(list)
    y_probs = []
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):

            # Set device
            X, y = X.to(device), y.to(device)

            # Forward pass
            conv_outputs_, logits = model(X)
            y_prob = F.softmax(logits, dim=1)

            # Save probabilities
            y_probs.extend(y_prob.cpu().numpy())
            for i, filter_size in enumerate(filter_sizes):
                conv_outputs[filter_size].extend(
                    conv_outputs_[i].cpu().numpy())

    return y_probs, conv_outputs


def predict(inputs, args, model, X_tokenizer, y_tokenizer):
    """Predict the class for a text using
    a trained model from an experiment."""

    # Preprocess
    texts = [sample['text'] for sample in inputs]
    preprocessed_texts = data.preprocess_texts(
        texts, lower=args.lower, filters=args.filters)

    # Create dataset
    X = np.array(X_tokenizer.texts_to_sequences(preprocessed_texts))
    y_filler = np.array([0]*len(X))
    dataset = data.TextDataset(
        X=X, y=y_filler, max_filter_size=max(args.filter_sizes))
    dataloader = dataset.create_dataloader(
        batch_size=args.batch_size)

    # Predict
    results = []
    y_prob, conv_outputs = predict_step(
        model=model, dataloader=dataloader,
        filter_sizes=args.filter_sizes, device='cpu')
    for index in range(len(X)):
        results.append({
            'raw_input': texts[index],
            'preprocessed_input': X_tokenizer.sequences_to_texts([X[index]])[0],
            'probabilities': get_probability_distribution(y_prob[index], y_tokenizer.classes),
            'top_n_grams': get_top_n_grams(tokens=preprocessed_texts[index].split(' '),
                                           conv_outputs={
                                               k: v[index] for k, v in conv_outputs.items()},
                                           filter_sizes=args.filter_sizes)})
    return results


if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--text', type=str,
                        required=True, help="text to predict")
    args = parser.parse_args()
    inputs = [{'text': args.text}]

    # Get best run
    best_run = utils.get_best_run(project="GokuMohandas/e2e-ml-app-pytorch",
                                  metric="test_loss", objective="minimize")

    # Load best run (if needed)
    best_run_dir = utils.load_run(run=best_run)

    # Get run components for prediction
    args, model, X_tokenizer, y_tokenizer = get_run_components(
        run_dir=best_run_dir)

    # Predict
    results = predict(inputs=inputs, args=args, model=model,
                      X_tokenizer=X_tokenizer, y_tokenizer=y_tokenizer)
    config.logger.info(json.dumps(results, indent=4, sort_keys=False))
