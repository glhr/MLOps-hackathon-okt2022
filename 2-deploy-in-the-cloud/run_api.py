#!pip install transformers sentencepiece datasets

from functools import cache
from typing import List
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import pandas as pd
import pickle
#from google.cloud import storage
from datasets import load_metric
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import re
from datasets import load_dataset, ClassLabel
import torch
import os
import pandas as pd
import datasets
from transformers import TextClassificationPipeline
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "python"
from google.cloud import storage
from pathlib import Path


app = FastAPI()

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)


def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc =  metric.compute(predictions=predictions, references=labels)
    return acc

def tok_func(x):
    tok_x = tokenizer(x["x"], padding=True, truncation=True)
    tok_x['label'] = x['label']
    return tok_x

def push_model():
    storage_client = storage.Client()
    bucket = storage_client.bucket("dataset-csv")
    for file in Path("model.pt").glob("*"):
        blob = bucket.blob(f'model.pt/{file.name}')
        print(f"uploading {file}")
        blob.upload_from_filename(file)

@cache
def pull_model():
    bucket_name = 'dataset-csv'
    prefix = 'model.pt/'
    dl_dir = 'model-dl.pt/'

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name=bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)  # Get list of files

    Path(dl_dir).mkdir(parents=True, exist_ok=True)
    for blob in blobs:
        filename = blob.name.split("/")[-1]
        blob.download_to_filename(dl_dir + filename)  # Download


class TrainRequest(BaseModel):
    dataset: str  # gs://path/to/dataset.csv
    model: str  # gs://path/to/model.pkl


@app.post("/train")
def train_model(req: TrainRequest):
    ds = pd.read_csv(req.dataset)
    key,ind = np.unique(ds['subreddit_id'],return_index=True)
    value = ds.subreddit[ind]
    label_map = {key: value for key,value in zip(key,value)}
    with open('label_map.pickle', 'wb') as handle:
        pickle.dump(label_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #save_model(label_map,'label_map')
    ds = ds[['post_title','subreddit_id']]
    ds = datasets.Dataset.from_pandas(ds)

    ds = ds.rename_column('subreddit_id', 'label')
    ds = ds.rename_column('post_title', 'x')
    num_labels = len(ds.unique('label'))
    ds = ds.train_test_split(test_size=0.2, shuffle=True)

    tok_ds = ds.map(tok_func, batched=True)
    print(tok_ds)
    #return
    ## Training parameters
    bs = 128
    epochs = 50
    lr = 5e-5
    args = TrainingArguments('model', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
                         evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
                         num_train_epochs=epochs, weight_decay=0.01, report_to='none')
    ##Tokenizer
    try:
        model = load_model(req.model)
    except:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model = model.to("cuda")
    trainer = Trainer(model, args, train_dataset=tok_ds['train'], eval_dataset=tok_ds['test'],
                  tokenizer=tokenizer, compute_metrics=compute_metrics)
    trainer.train();
    trainer.save_model("model.pt")
    push_model()



class PredictRequest(BaseModel):
    model: str  # gs://path/to/model.pkl
    sample: str


@app.post("/predict",response_class=PlainTextResponse)
def predict(req: PredictRequest):

    with open('label_map.pickle', 'rb') as handle:
        label_map = pickle.load(handle)
    num_labels = len(label_map)

    if not Path("model-dl.pt/pytorch_model.bin").is_file():
        pull_model()

    model = AutoModelForSequenceClassification.from_pretrained('model-dl.pt',num_labels=num_labels)
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=3)
    predictions = pipe(req.sample)[0]
    print(f"loaded label_map",label_map)
    #return predictions
    result_dict = {label_map[int(pred['label'].replace("LABEL_",""))]: float(pred['score']) for pred in predictions}
    result_str = "\n".join([f"{l} ({s})" for l,s in result_dict.items()])
    return f"Consider posting this thoughtful post on:\n{result_str}"
