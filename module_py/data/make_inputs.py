import os, sys
import json
import pandas as pd
from transformers import ElectraTokenizer
from sklearn.model_selection import train_test_split
from .datasets import (
    make_seed_dataset,
    batch_encode,
    make_features,
    make_inputs,
    make_dataloader,
)
# 절대 경로 참조
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.utils import args, path, set_seed


# seed 고정
set_seed(args)

# train, dev json 로드
train_json = "klue-sts-v1.1_train.json"
dev_json = "klue-sts-v1.1_dev.json"

with open(f"{path['json']}/{train_json}", "r") as train_json:
    train = json.load(train_json)

with open(f"{path['json']}/{dev_json}", "r") as dev_json:
    dev = json.load(dev_json)

# json to csv
train = pd.json_normalize(train).iloc[:, 0:7] # shape = (11668, 7)
dev = pd.json_normalize(dev).iloc[:, 0:7]     # shape = (519, 7)

# 중복값 확인
tr_dp = train.duplicated(subset=['source','sentence1','sentence2','labels.label','labels.real-label','labels.binary-label']).sum()
dv_dp = dev.duplicated(subset=['source','sentence1','sentence2','labels.label','labels.real-label','labels.binary-label']).sum()

# 중복값 제거
if tr_dp > 0:
    train.drop_duplicates(subset=['source','sentence1','sentence2','labels.label','labels.real-label','labels.binary-label'], inplace=True)
if dv_dp > 0:
    dev.drop_duplicates(subset=['source','sentence1','sentence2','labels.label','labels.real-label','labels.binary-label'], inplace=True)

# 컬럼명 변경
train.columns = ['guid', 'src', 'sent1', 'sent2', 'f_label', 'real_f_label', 'b_label']
dev.columns = ['guid', 'src', 'sent1', 'sent2', 'f_label', 'real_f_label', 'b_label']

# train, val 분할
train, val = train_test_split(train, test_size=0.1, random_state=args['seed'])

# train, val 데이터수 확인
print(f"훈련데이터 개수 : {len(train)}")
print(f"검증데이터 개수 : {len(val)}")


""" make_seed_dataset returns...
- sent_paris : (문장1, 문장2) pair
- labels     : (float) label
- b_label    : (binary) label
"""
# train dataset
t_sent_pairs, t_labels, t_b_labels = make_seed_dataset(train, 'sent1', 'sent2', 'real_f_label', 'b_label')
# validation dataset
v_sent_pairs, v_labels, v_b_labels = make_seed_dataset(val, 'sent1', 'sent2', 'real_f_label', 'b_label')
# dev dataset
d_sent_pairs, d_labels, d_b_labels = make_seed_dataset(dev, 'sent1', 'sent2', 'real_f_label', 'b_label')

"""batch_encode returns...
{'input_ids':[...],
 'attention_mask':[...],
 'token_type_ids':[...]}
"""
tokenizer = ElectraTokenizer.from_pretrained(args['model_name_or_path'], do_lower_case=False)
# Do encode
t_batch_encoding = batch_encode(tokenizer, t_sent_pairs)
v_batch_encoding = batch_encode(tokenizer, v_sent_pairs) 
d_batch_encoding = batch_encode(tokenizer, d_sent_pairs)

"""make_features returns...
Labels are included in the returns from batch_encode function above.
{'input_ids':[...],
 'attention_mask':[...],
 'token_type_ids':[...],
 'label':[.]}
"""
# Get features 
t_features = make_features(t_sent_pairs, t_batch_encoding, t_labels)
v_features = make_features(v_sent_pairs, v_batch_encoding, v_labels)
d_features = make_features(d_sent_pairs, d_batch_encoding, d_labels)

"""make_inputs returns...
Tensorized dataset
- TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
"""
# make inputs
t_dataset = make_inputs(t_features)
v_dataset = make_inputs(v_features)
d_dataset = make_inputs(d_features)
print(f"inputs 개수: train({len(t_dataset)})\tvalid({len(v_dataset)})\tdev({len(d_dataset)})")

# DataLoader
train_dataloader = make_dataloader(t_dataset, mode='train')
valid_dataloader = make_dataloader(v_dataset, mode='valid')
dev_dataloader = make_dataloader(d_dataset, mode='dev')
print("train, valid, dev are all on each dataloader !")