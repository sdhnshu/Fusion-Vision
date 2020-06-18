import os
import sys
sys.path.append(".")
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from io import BytesIO
import json
import numpy as np
import shutil
import ssl
from urllib.request import urlopen
import wandb
from zipfile import ZipFile

from text_classification import config

# Read data from URLs
ssl._create_default_https_context = ssl._create_unverified_context


def create_dirs(dirpath):
    """Creating directories."""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def load_json(filepath):
    """Load a json file."""
    with open(filepath, "r") as fp:
        json_obj = json.load(fp)
    return json_obj


def save_dict(d, filepath):
    """Save dict to a json file."""
    with open(filepath, 'w') as fp:
        json.dump(d, indent=2, sort_keys=False, fp=fp)


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""
    @wraps(f)
    def wrap(*args, **kwargs):
        results = f(*args, **kwargs)

        # Construct response
        response = {
            'message': results['message'],
            'method': request.method,
            'status-code': results['status-code'],
            'timestamp': datetime.now().isoformat(),
            'url': request.url,
        }

        # Add data
        if results['status-code'] == HTTPStatus.OK:
            response['data'] = results['data']

        return response

    return wrap


def load_glove_embeddings(embeddings_file):
    """Load embeddings from a file."""
    embeddings = {}
    with open(embeddings_file, "r") as fp:
        for index, line in enumerate(fp):
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings[word] = embedding
    return embeddings


def make_embeddings_matrix(embeddings, token_to_index, embedding_dim):
    """Create embeddings matrix to use in Embedding layer."""
    embedding_matrix = np.zeros((len(token_to_index)+1, embedding_dim))
    for word, i in token_to_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def get_best_run(project, metric, objective):
    # Get all runs
    api = wandb.Api()
    runs = api.runs(project)

    # Define objective
    if objective == 'maximize':
        best_metric_value = np.NINF
    elif objective == 'minimize':
        best_metric_value = np.inf

    # Get best run based on metric
    best_run = None
    for run in runs:
        if run.state == "finished":
            metric_value = run.summary[metric]
            if objective == 'maximize':
                if metric_value > best_metric_value:
                    best_run = run
                    best_metric_value = metric_value
            else:
                if metric_value < best_metric_value:
                    best_run = run
                    best_metric_value = metric_value

    return best_run


def load_run(run):
    run_dir = os.path.join(
        config.BASE_DIR, '/'.join(run.summary['run_dir'].split('/')[-2:]))

    # Create run dir if it doesn't exist
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    else:
        return run_dir

    # Load run files (if it exists, nothing happens)
    for file in run.files():
        file.download(replace=False, root=run_dir)

    return run_dir


if __name__ == '__main__':
    # Unzip and write embeddings (may take ~3-5 minutes)
    resp = urlopen('http://nlp.stanford.edu/data/glove.6B.zip')
    embeddings_dir = os.path.join(config.BASE_DIR, 'embeddings')
    create_dirs(embeddings_dir)
    with ZipFile(BytesIO(resp.read()), 'r') as zr:
        zr.extractall(embeddings_dir)
