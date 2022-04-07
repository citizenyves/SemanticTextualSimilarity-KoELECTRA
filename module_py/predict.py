import os, sys
import numpy as np
import torch

# 절대 경로 참조
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.utils import args, path, set_seed, compute_metrics
from modeling.utils import Initializer
from modeling.predict import predict
from data.make_inputs import dev, dev_dataloader

# device type
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"# available GPUs : {torch.cuda.device_count()}")
    print(f"GPU name : {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
print(device)

# seed 고정
set_seed(args)

# Load model
checkpoint = torch.load(os.path.join(path['ckpt'], 'model.ckpt.19'), map_location=torch.device(device))

# Initialize model & optimizer & scheduler
initializer = Initializer(args)
model = initializer.model_initializer()
# optimizer, scheduler, _ = initializer.opt_sch_initializer(model, train_dataloader)

# model_state_dict
model.load_state_dict(checkpoint["model_state_dict"])

################################################
#                                              #
#                   Predict                    #
#                                              #
################################################
preds, out_label_ids, loss = predict(args, model, device, dev_dataloader, 'test')


################################################
#                                              #
#                  Evaluate                    #
#                                              #
################################################
#              1.Make Dataframe                #
################################################

# pred label (float)
pred_real_label = [pred for pred in preds]

# pred label (binary) 생성 (원 데이터 기준 3.000점 이상일 시 1, 미만일 시 0)
pred_bi_label = [1 if i >= 3 else 0 for i in preds]

# dev_set_score (최종) 데이터프레임 생성
dev['pred_real_f_label'] = pred_real_label
dev['pred_b_label'] = pred_bi_label

dev_set_score = dev[['guid', 'real_f_label', 'b_label', 
                     'pred_real_f_label', 'pred_b_label']].rename(columns={'real_f_label':'true_real_label',
                                                                                                              'b_label':'true_binary_label',
                                                                                                              'pred_real_f_label':'predict_real_label',
                                                                                                              'pred_b_label':'predict_binary_label'})
print(dev_set_score.head(5))

# Export CSV
fname = 'dev_set_score.csv'
dev_set_score.to_csv(f'/Users/mac/project/STS_KoELECTRA/data/{fname}')


################################################
#            2.pearsonr & spearmanr            #
################################################
corr = compute_metrics("pearson_and_spearman", dev_set_score['true_real_label'], dev_set_score['predict_real_label'])
print(corr)
print('\n')

################################################
#                  3.F1 score                  #
################################################
f1 = compute_metrics("f1_pre_rec",dev_set_score['true_binary_label'], dev_set_score['predict_binary_label'])['f1']
precision = compute_metrics("f1_pre_rec",dev_set_score['true_binary_label'], dev_set_score['predict_binary_label'])['precision']
recall = compute_metrics("f1_pre_rec",dev_set_score['true_binary_label'], dev_set_score['predict_binary_label'])['recall']  
print(f"f1 score  : {f1}")
print(f"precision : {precision}")
print(f"recall    : {recall}")