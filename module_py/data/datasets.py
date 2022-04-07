import os, sys
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# 절대 경로 참조
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.utils import args


def make_seed_dataset(df, col1, col2, f_label, b_label):
    # sentence pairs
    sent_pairs = [(sent1, sent2) for sent1, sent2 in zip(df[col1], df[col2])]
    
    # labels(float)
    labels = [label for label in df[f_label]]

    # labels(binary)
    b_labels = [label for label in df[b_label]]
    
    return sent_pairs, labels, b_labels


def batch_encode(tokenizer, sent_pairs):
    batch_encoding = tokenizer.batch_encode_plus(
                              [(pairs[0], pairs[1]) for pairs in tqdm(sent_pairs, total=len(sent_pairs))],
                              max_length=128,
                              padding="max_length",    
                              add_special_tokens=True,
                              truncation=True,
                              )
    return batch_encoding


def make_features(sent_pairs, batch_encoding, labels):
    features = []
    for i in tqdm(range(len(sent_pairs)), total=len(sent_pairs)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = [0] * len(inputs["input_ids"])

        feature = {'input_ids':inputs['input_ids'],
                  'attention_mask':inputs['attention_mask'],
                  'token_type_ids':inputs['token_type_ids'],
                  'label':labels[i]
                  }
        
        features.append(feature)
    
    return features


def make_inputs(features, output_mode:str="regression"):
    assert output_mode in ["classification", "regression"]

    # inputs
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)

    # label
    if output_mode == "classification":
        all_labels = torch.tensor([f['label'] for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f['label'] for f in features], dtype=torch.float)
    else:
        raise AssertionError(output_mode)

    # dataset
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return dataset


def make_dataloader(dataset, mode:str):
    assert mode in ['train', 'valid', 'dev']

    if mode == "train":
        dataloader = DataLoader(dataset = dataset, # (input_ids, attention_mask, token_type_ids, labels)
                                sampler = RandomSampler(dataset), 
                                batch_size = args['train_batch_size'])
    elif mode == "valid":
        dataloader = DataLoader(dataset = dataset, 
                                sampler = RandomSampler(dataset), 
                                batch_size = args['eval_batch_size'])
    elif mode == "dev":
        dataloader = DataLoader(dataset = dataset, 
                                sampler = SequentialSampler(dataset), 
                                batch_size = args['eval_batch_size'])
    else:
        raise AssertionError(mode)

    return dataloader

# EDA PART #
def describe_sentence_statistic(dataset, tokenizer):
    len_sent1 = []
    len_sent2 = []

    for sent1, sent2 in tqdm(zip(dataset['sent1'], dataset['sent2']), total=len(dataset)):
        len_sent1.append(len(tokenizer.encode(sent1)))
        len_sent2.append(len(tokenizer.encode(sent2)))

    describe = pd.DataFrame(data={'sent 1':len_sent1, 'sent 2':len_sent2})\
                 .describe(percentiles=[.05, .10, .25, .5, .75, .90, .95, .99, .999, .9999,.99999])
    
    return len_sent1, len_sent2, describe

def length_visual(len_sent1, len_sent2, mode='train'):
    print(f"({mode}set) just visualized !")
    plt.rcParams["font.family"] = "serif"

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white", dpi=72) 
    x = len_sent1
    y = len_sent2

    x_bins = np.linspace(min(x), max(x), 100)
    y_bins = np.linspace(min(y), max(y), 100)  

    plt.hist2d(x, y, bins=[x_bins, y_bins], cmap="PiYG", norm=matplotlib.colors.LogNorm()) 
    cbar = plt.colorbar() 
    cbar.set_label("Counts", fontsize=14)

    plt.xlim([0, 120])
    plt.ylim([0, 120])
    plt.xlabel("Length of Sentence1", fontsize=14)
    plt.ylabel("Length of Sentence2", fontsize=14)

    ax.tick_params(axis="both", direction="out")  
    plt.grid(True, lw=0.15)

    plt.tight_layout()  
    # plt.savefig("2.png", dpi=300)
    plt.show()