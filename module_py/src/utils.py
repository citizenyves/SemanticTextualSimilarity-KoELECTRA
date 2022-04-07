import numpy as np
import random
import torch
import logging
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics as sklearn_metrics


args = {
  "task": "korsts",
  "evaluate_test_during_training": False,
  "do_lower_case": False,
  "do_train": True,
  "max_seq_len": 128,
  "num_train_epochs": 3,
  "weight_decay": 0.0,   # default = 1e-2
  "gradient_accumulation_steps": 1,
  "betas": (0.9, 0.999), # default
  "adam_epsilon": 1e-8,  # default
  "warmup_proportion": 0.2,
  "max_steps": -1,
  "max_grad_norm": 1.0,
  "no_cuda": False,
  "model_type": "koelectra-base-v3",
  "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
  "seed": 42,
  "train_batch_size": 32,
  "eval_batch_size": 64,
  "learning_rate": 5e-5,
  "output_mode":"regression"
}

path = {
    "json": "/Users/mac/project/STS_KoELECTRA/data",
    "export_data": "/Users/mac/project/STS_KoELECTRA/data",
    "logdir": "/Users/mac/project/STS_KoELECTRA/logdir",
    "ckpt": "/Users/mac/project/STS_KoELECTRA/model",
}

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def set_seed(args):
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])

def pearson_and_spearman(labels, preds):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def f1_pre_rec(labels, preds):
    return {
        "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
        "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
        "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
    }

def compute_metrics(metric_name, labels, preds):
    assert len(preds) == len(labels)
    if metric_name == "pearson_and_spearman":
        return pearson_and_spearman(labels, preds)
    elif metric_name == "f1_pre_rec":
        return f1_pre_rec(labels, preds)
    else:
        raise KeyError(metric_name)


