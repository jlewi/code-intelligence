"""Build a classification model with hugging face."""

import fire
import logging
import torch
from google.cloud import storage
import os
from code_intelligence import gcs_util
import io
from sklearn import model_selection
import numpy as np
import datetime
import pandas as pd

from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import json

DEFAULT_FILE_LIST = "gs://issue-label-bot-dev_automl/automl_TCN8830229559715561472/dataset_201103_151334.csv"

def split_labels(x):
  if isinstance(x, float):
    return []

  return [i.strip() for i in x.split(",")]


def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average="micro")

class HFModel:
  @staticmethod
  def load_data(files_list=DEFAULT_FILE_LIST):
    storage_client = storage.Client()
    bucket_name, obj_path = gcs_util.split_gcs_uri(files_list)
    b = storage_client.bucket(bucket_name)
    o = b.blob(obj_path)

    csv_contents = o.download_as_string()
    files = pd.read_csv(io.BytesIO(csv_contents), names=["garbage", "file", "labels"])

    files["text"] = ""

    # Compute the number of items related to progress
    percent_interval = 1
    progress = np.floor(files.shape[0] * percent_interval / 100.0)

    # TODO(jlewi): Reading from GCS may not be efficient. The GCS data is obtained in automl.ipynb by creating a dataframe from BigQuery
    # It might be faster just to read the data directly from BigQuery and if necessary store the dataframe as an hdf5 file
    start = datetime.datetime.now()

    for index in range(files.shape[0]):
      if index % progress == 0:
        percent = index/files.shape[0] * 100
        elapsed = datetime.datetime.now() - start
        print(f"Percent done {percent}; Processing {index+1} of {files.shape[0]}; elapsed {elapsed}")
      bucket_name, obj_path = gcs_util.split_gcs_uri(files_list)
      b = storage_client.bucket(bucket_name)
      o = b.blob(obj_path)
      files.at[index, "text"] = o.download_as_string()

    files["labels"] = files["labels_raw"].apply(split_labels)

    return files

  @staticmethod
  def train(use_cuda=True, max_issues=0):
    """Train a model using hugging face.

       args:
        max_issues: If supplied use at most this many issues
          this is purely to subsample the data to make it run quickly.

       This uses the simpletransformers library
       https://github.com/ThilinaRajapakse/simpletransformers/tree/master/simpletransformers/classification
       which is built ontop of HuggingFace

       This code is based on the vscode issue label model
       https://github.com/microsoft/vscode-github-triage-actions/blob/master/classifier-deep/train/vm-filesystem/classifier/generateModels.py)"""

    files = HFModel.load_data()

    if max_issues:
      logging.info(f"Truncating files to first {max_issues} rows")
      files = files[:max_issues]

    # split the data into train and test sets
    train_files, val_files = model_selection.train_test_split(files, test_size=.2)

    # Generate a unique list of labels
    target_labels = functools.reduce(np.union1d, files["labels"].values)

    category = "area"

    model_args = ClassificationArgs(
        output_dir=category + "_model",
        best_model_dir=category + "_model_best",
        overwrite_output_dir=True,
        train_batch_size=16,
        eval_batch_size=32,
        max_seq_length=256,
        num_train_epochs=2,
        save_model_every_epoch=False,
        save_eval_checkpoints=False,
    )

    # Create a ClassificationModel
    model = ClassificationModel(
        # TODO(jlewi): bert finetuned isn't found.
        # So I used what I found in https://github.com/ThilinaRajapakse/simpletransformers/blob/master/examples/text_classification/multilabel_classification.py
        # "bert", "finetuned",
        "roberta", "roberta-base",
        num_labels=len(target_labels), args=model_args,
        use_cuda=use_cuda,
    )

    # Train the model
    # TODO(jlewi): Save the model to GCS
    model.train_model(
        train_df, eval_df=test_df, output_dir=category + "_model/checkpoints",
    )

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(
        test_df,
        output_dir=category + "_model/eval",
        f1=f1_multiclass,
        acc=accuracy_score,
    )

    # TODO(jlewi): Save the model to GCS
    with open(os.path.join(category + "_model", "target_names.json"), "w") as f:
        json.dump(target_names, f)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO,
                      format=('%(levelname)s|%(asctime)s'
                              '|%(message)s|%(pathname)s|%(lineno)d|'),
                      datefmt='%Y-%m-%dT%H:%M:%S',
                      )

  fire.Fire(HFModel)
