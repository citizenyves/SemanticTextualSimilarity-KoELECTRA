import os, sys
import json
import pandas as pd
from transformers import ElectraTokenizer
from .datasets import describe_sentence_statistic, length_visual
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


# info
print("######### info #########")
print(train.info())
print(dev.info())
print("\n")

# 결측값 확인
print("######### 결측값 확인 #########")
print(train.isnull().sum())
print(dev.isnull().sum())
print("\n")

# 중복값 확인
print("######### 중복값 확인 #########")
tr_dp = train.duplicated(subset=['source','sentence1','sentence2','labels.label','labels.real-label','labels.binary-label']).sum()
dv_dp = dev.duplicated(subset=['source','sentence1','sentence2','labels.label','labels.real-label','labels.binary-label']).sum()
print(f"train set 중복값: {tr_dp}\t dev set 중복값: {dv_dp}")

# 중복값 제거
if tr_dp > 0:
    train.drop_duplicates(subset=['source','sentence1','sentence2','labels.label','labels.real-label','labels.binary-label'], inplace=True)
    print(f">>> DUPLICATED VALUES in the train set({tr_dp}EA) ARE JUST REMOVED !!")
if dv_dp > 0:
    dev.drop_duplicates(subset=['source','sentence1','sentence2','labels.label','labels.real-label','labels.binary-label'], inplace=True)
    print(f">>> DUPLICATED VALUES in the dev set({dv_dp}EA) ARE JUST REMOVED !!")
print("\n")

# 컬럼명 변경
train.columns = ['guid', 'src', 'sent1', 'sent2', 'f_label', 'real_f_label', 'b_label']
dev.columns = ['guid', 'src', 'sent1', 'sent2', 'f_label', 'real_f_label', 'b_label']

# 문장당 토큰수에 대한 통계치 확인
tokenizer = ElectraTokenizer.from_pretrained(args['model_name_or_path'], do_lower_case=False)
print("ElectraTokenizer is just called away !")
print("\n")
print("Sentence statistic is in progress...")
tr_len_sent1, tr_len_sent2, train_describe = describe_sentence_statistic(train, tokenizer)
dv_len_sent1, dv_len_sent2, dev_describe = describe_sentence_statistic(dev, tokenizer)
print("\n")
print("****** SENTENCE STATISTIC ******")
print("********** TRAIN SET ***********")
print(train_describe)
print("*********** DEV SET ************")
print(dev_describe)

# trainset 문장 통계 시각화
length_visual(tr_len_sent1, tr_len_sent2)
# devset 문장 통계 시각화
length_visual(dv_len_sent1, dv_len_sent2, mode='dev')