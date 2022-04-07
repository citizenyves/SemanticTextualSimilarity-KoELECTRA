import os, sys
import torch
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
# 절대 경로 참조
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.processor import (
    CONFIG_CLASSES, MODEL_FOR_SEQUENCE_CLASSIFICATION,
    seq_cls_tasks_num_labels,
)

# 체크포인트 저장 함수
def save_checkpoint(path, model, optimizer, scheduler, epoch, loss):
    file_name = f'{path}/model.ckpt.{epoch}'
    torch.save(
        {
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict(),
          'loss' : loss
        },
        file_name
    )

    print(f"Saving epoch {epoch} checkpoint at {file_name}")

# 모델, 옵티마이저, 스케줄러 초기화 클래스
class Initializer():
    def __init__(self, args):
        self.args = args
        self.config = CONFIG_CLASSES[args['model_type']].from_pretrained(
                args['model_name_or_path'],
                num_labels=seq_cls_tasks_num_labels[args['task']]
            )
        self.model = MODEL_FOR_SEQUENCE_CLASSIFICATION[args['model_type']].from_pretrained(
            args['model_name_or_path'],
            config=self.config
        )

    def model_initializer(self):
        """
        모델을 초기화한 후 반환
        """

        return self.model


    def opt_sch_initializer(self, model, train_dataloader):
        """
        옵티마이저, 스케쥴러를 초기화한 후 반환
        """

        # optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args['weight_decay']}, #weight_decay = 0.0
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args['learning_rate'],
                          eps=self.args['adam_epsilon'])
        
        # scheduler
        if self.args['max_steps'] > 0: 
            t_total = self.args['max_steps']
            self.args['num_train_epochs'] = self.args['max_steps'] // (len(train_dataloader) // self.args['gradient_accumulation_steps']) + 1
        else:
            t_total = len(train_dataloader) // self.args['gradient_accumulation_steps'] * self.args['num_train_epochs']

        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=int(t_total * self.args['warmup_proportion']), #0 
                                                    num_training_steps=t_total)
        
        return optimizer, scheduler, t_total