# Creating an End-to-End ML Application w/ <img src="https://raw.githubusercontent.com/madewithml/images/master/images/pytorch.png" width="25rem"> PyTorch

🚀 This project was created using the Made With ML [boilerplate](https://github.com/madewithml/boilerplate) template. Check it out to start creating your own ML applications.

## Overview
- **Why do we need to build end-to-end applications?**
    - By building e2e applications, you ensure that your code is organized, tested, testable / interactive and easy to scale-up / assimilate with larger pipelines.
    - If you're someone in industry and are looking to showcase your work to future employers, it's no longer enough to just have code on Jupyter notebooks. ML is just another tool and you need to show that you can use it in conjunction with all the other software engineering disciplines (frontend, backend, devops, etc.). The perfect way to do this is to create end-to-end applications that utilize all these different facets.
- **What are the components of an end-to-end ML application?**
    1. Basic experimentation in Jupyter notebooks.
        - We aren't going to completely dismiss notebooks because they're still great tool to iterate quickly. Check out the notebook for our task here → [notebook](https://github.com/madewithml/e2e-ml-app-pytorch/blob/master/notebook.ipynb)
    2. Moving our code from notebooks to organized scripts.
        - Once we did some basic development (on downsized datasets), we want to move our code to scripts to reduce technical debt. We'll create functions and classes for different parts of the pipeline (data, model, train, etc.) so we can easily make them robust for different circumstances.
        - We used our own [boilerplate](https://github.com/madewithml/boilerplate) to organize our code before moving any of the code from our notebook.
    3. Proper logging and testing for you code.
        - Log key events (preprocessing, training performance, etc.) using the built-in [logging](https://docs.python.org/2/howto/logging.html) library. Also use logging to see new inputs and outputs during prediction to catch issues, etc.
        - You also need to properly test your code. You will add and update your functions and their tests over time but it's important to at least start testing crucial pieces of your code from the beginning. These typically include sanity checks with preprocessing and modeling functions to catch issues early. There are many options for testing Python code but we'll use [pytest](https://docs.pytest.org/en/stable/) here.
    4. Experiment tracking.
        - We use [Weights and Biases](https://wandb.com) (WandB), where you can easily track all the metrics of your experiment, config files, performance details, etc. for free. Check out the [Dashboards page](https://www.wandb.com/experiment-tracking) for an overview and tutorials.
        - When you're developing your models, start with simple approaches first and then slowly add complexity. You should clearly document (README, articles and [WandB reports](https://www.wandb.com/articles/workspaces-tables-reports-oh-my)) and save your progression from simple to more complex models so your audience can see the improvements. The ability to write well and document your thinking process is a core skill to have in research and industry.
        - WandB also has free tools for hyperparameter tuning ([Sweeps](https://www.wandb.com/sweeps)) and for data/pipeline/model management ([Artifacts](https://www.wandb.com/artifacts)).
    5. Robust prediction pipelines.
        - When you actually deploy an ML application for the real world to use, we don't just look at the softmax scores.
        - Before even doing any forward pass, we need to analyze the input and deem if it's within the manifold of the training data. If it's something new (or adversarial) we shouldn't send it down the ML pipeline because the results cannot be trusted.
        - During processes like proprocessing, we need to constantly observe what the model received. For example, if the input has a bunch of unknown tokens than we need to flag the prediction because it may not be reliable.
        - After the forward pass we need to do tests on the model's output as well. If the predicted class has a mediocre test set performance, then we need the class probability to be above some critical threshold. Similarly we can relax the threshold for classes where we do exceptionally well.
    6. Wrap your model as an API.
        - Now we start to modularize larger operations (single/batch predict, get experiment details, etc.) so others can use our application without having to execute granular code. There are many options for this like [Flask](https://flask.palletsprojects.com/en/1.1.x/), [Django](https://www.djangoproject.com/), [FastAPI](https://fastapi.tiangolo.com/), etc. but we'll use FastAPI for the ease and [performance](https://fastapi.tiangolo.com/#performance) boost.
        - We can also use a Dockerfile to create a [Docker](https://towardsdatascience.com/how-docker-can-help-you-become-a-more-effective-data-scientist-7fc048ef91d5) image that runs our API. This is a great way to package our entire application to scale it (horizontally and vertically) depending on requirements and usage.
    7. Create an interactive frontend for your application.
        - The best way to showcase your work is to let others easily play with it. We'll be using [Streamlit](https://www.streamlit.io/) to very quickly create an interactive medium for our application and use [Heroku](https://heroku.com/) to serve it (1000 hours of usage per month).
        - This is also a great skill to have because in industry you'll need to create this to show key stakeholders and great to have in documentation as well.

## Set up
```
virtualenv -p python3.6 venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch==1.4.0
```

## Download embeddings
```bash
python text_classification/utils.py
```

## Training
```bash
python text_classification/train.py \
    --data-url https://raw.githubusercontent.com/madewithml/lessons/master/data/news.csv --lower --shuffle --use-glove
```

## Endpoints
```bash
uvicorn text_classification.app:app --host 0.0.0.0 --port 5000 --reload
GOTO: http://localhost:5000/docs
```

## Prediction
### Scripts
```bash
python text_classification/predict.py --text 'The Canadian government officials proposed the new federal law.'
```

### cURL
```
curl "http://localhost:5000/predict" \
    -X POST -H "Content-Type: application/json" \
    -d '{
            "inputs":[
                {
                    "text":"The Wimbledon tennis tournament starts next week!"
                },
                {
                    "text":"The Canadian government officials proposed the new federal law."
                }
            ]
        }' | json_pp
```

### Requests
```python
import json
import requests

headers = {
    'Content-Type': 'application/json',
}

data = {
    "experiment_id": "latest",
    "inputs": [
        {
            "text": "The Wimbledon tennis tournament starts next week!"
        },
        {
            "text": "The Canadian minister signed in the new federal law."
        }
    ]
}

response = requests.post('http://0.0.0.0:5000/predict',
                         headers=headers, data=json.dumps(data))
results = json.loads(response.text)
print (json.dumps(results, indent=2, sort_keys=False))
```

## Streamlit
```bash
streamlit run text_classification/streamlit.py
GOTO: http://localhost:8501
```

## Tests
```bash
pytest
```

## Docker
1. Build image
```bash
docker build -t text-classification:latest -f Dockerfile .
```
2. Run container
```bash
docker run -d -p 5000:5000 -p 6006:6006 --name text-classification text-classification:latest
```

## Heroku
```
Set `WANDB_API_KEY` as an environment variable.
```

## Directory structure
```
text-classification/
├── datasets/                           - datasets
├── logs/                               - directory of log files
|   ├── errors/                           - error log
|   └── info/                             - info log
├── tests/                              - unit tests
├── text_classification/                - ml scripts
|   ├── app.py                            - app endpoints
|   ├── config.py                         - configuration
|   ├── data.py                           - data processing
|   ├── models.py                         - model architectures
|   ├── predict.py                        - prediction script
|   ├── streamlit.py                      - streamlit app
|   ├── train.py                          - training script
|   └── utils.py                          - load embeddings
├── wandb/                              - wandb experiment runs
├── .dockerignore                       - files to ignore on docker
├── .gitignore                          - files to ignore on git
├── CODE_OF_CONDUCT.md                  - code of conduct
├── CODEOWNERS                          - code owner assignments
├── CONTRIBUTING.md                     - contributing guidelines
├── Dockerfile                          - dockerfile to containerize app
├── LICENSE                             - license description
├── logging.json                        - logger configuration
├── Procfile                            - process script for Heroku
├── README.md                           - this README
├── requirements.txt                    - requirementss
├── setup.sh                            - streamlit setup for Heroku
└── sweeps.yaml                         - hyperparameter wandb sweeps config
```

## Overfit to small subset
```
python text_classification/train.py \
    --data-url https://raw.githubusercontent.com/madewithml/lessons/master/data/news.csv --lower --shuffle --data-size 0.1 --num-epochs 3
```

## Experiments
1. Random, unfrozen, embeddings
```
python text_classification/train.py \
    --data-url https://raw.githubusercontent.com/madewithml/lessons/master/data/news.csv --lower --shuffle
```
2. GloVe, frozen, embeddings
```
python text_classification/train.py \
    --data-url https://raw.githubusercontent.com/madewithml/lessons/master/data/news.csv --lower --shuffle --use-glove --freeze-embeddings
```
3. GloVe, unfrozen, embeddings
```
python text_classification/train.py \
    --data-url https://raw.githubusercontent.com/madewithml/lessons/master/data/news.csv --lower --shuffle --use-glove
```

## Next steps
End-to-end topics that will be covered in subsequent lessons.
- Utilizing wrappers like [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to structure the modeling even more while getting some very useful utility.
- Data / model version control ([Artifacts](https://www.wandb.com/artifacts), [DVC](https://dvc.org/), [MLFlow](https://mlflow.org/), etc.)
- Experiment tracking options ([MLFlow](https://mlflow.org/), [KubeFlow](https://www.kubeflow.org/), [WandB](https://www.wandb.com/), [Comet](https://www.comet.ml/site/), [Neptune](https://neptune.ai/), etc)
- Hyperparameter tuning options ([Optuna](https://optuna.org/), [Hyperopt](https://github.com/hyperopt/hyperopt), [Sweeps](https://www.wandb.com/sweeps))
- Multi-process data loading
- Dealing with imbalanced datasets
- Distributed training for much larger models
- GitHub actions for automatic testing during commits
- Prediction fail safe techniques (input analysis, class-specific thresholds, etc.)

## Helpful docker commands
• Build image
```
docker build -t madewithml:latest -f Dockerfile .
```

• Run container if using `CMD ["python", "app.py"]` or `ENTRYPOINT [ "/bin/sh", "entrypoint.sh"]`
```
docker run -p 5000:5000 --name madewithml madewithml:latest
```

• Get inside container if using `CMD ["/bin/bash"]`
```
docker run -p 5000:5000 -it madewithml /bin/bash
```

• Run container with mounted volume
```
docker run -p 5000:5000 -v $PWD:/root/madewithml/ --name madewithml madewithml:latest
```

• Other flags
```
-d: detached
-ti: interative terminal
```

• Clean up
```
docker stop $(docker ps -a -q)     # stop all containers
docker rm $(docker ps -a -q)       # remove all containers
docker rmi $(docker images -a -q)  # remove all images
```

