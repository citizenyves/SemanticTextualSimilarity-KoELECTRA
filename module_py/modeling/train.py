import os, sys
import logger
import logging
import numpy as np
import torch
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from fastprogress.fastprogress import master_bar, progress_bar
from .utils import save_checkpoint
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.utils import args, path, compute_metrics
from src.processor import seq_cls_output_modes
from data.make_inputs import t_dataset, v_dataset


def train(args, model, device, writer, train_dataloader, valid_dataloader=None):
    # 1 epoch 주기로 train, val loss를 저장하는 리스트
    avg_tr_loss_li = []
    avg_val_loss_li = []

    # train total steps
    if args['max_steps'] > 0: 
        t_total = args['max_steps']
        args['num_train_epochs'] = args['max_steps'] // (len(train_dataloader) // args['gradient_accumulation_steps']) + 1
    else:
        t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']

    # optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args['learning_rate'],
                      betas=args['betas'],
                      eps=args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=int(t_total * args['warmup_proportion']), 
                                                num_training_steps=t_total)

    # Load optimizer and scheduler states
    if os.path.isfile(os.path.join(args['model_name_or_path'], "optimizer.pt")) and os.path.isfile(
            os.path.join(args['model_name_or_path'], "scheduler.pt")
    ):
        optimizer.load_state_dict(torch.load(os.path.join(args['model_name_or_path'], "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args['model_name_or_path'], "scheduler.pt")))

    # logger info
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(t_dataset))
    logging.info("  Num Epochs = %d", args['num_train_epochs'])
    logging.info("  Total train batch size = %d", args['train_batch_size'])
    logging.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logging.info("  Total optimization steps = %d", t_total)

    # step and loss 초기화
    global_step, batch_step = 0, 0
    tr_loss, batch_loss = 0.0, 0.0

    # zero_grad
    model.zero_grad()

    # Train!
    mb = master_bar(range(int(args['num_train_epochs'])))
    for epoch in tqdm(mb):
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            batch_step+=1
            
            # train mode
            model.train()
            
            # inputs to device
            batch = tuple(item.to(device) for item in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]
            }
            
            # forward
            outputs = model(**inputs)

            # loss 
            loss = outputs[0]
            # loss depending on the gradient_accumulation_steps
            if args['gradient_accumulation_steps'] > 1: 
                loss = loss / args['gradient_accumulation_steps']

            # backward
            loss.backward()

            # batch_loss & (total)loss
            batch_loss += loss.item()
            tr_loss += loss.item()
            
            if (step + 1) % args['gradient_accumulation_steps'] == 0 or (
                    len(train_dataloader) <= args['gradient_accumulation_steps']
                    and (step + 1) == len(train_dataloader)
            ):  # gradient clipping (max_norm = 1)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                               max_norm = args['max_grad_norm'])
                
                # optimizer & scheduler 업데이트
                optimizer.step()
                scheduler.step()

                # batch마다 모델이 갖고 있는 기존 gradient를 초기화
                model.zero_grad()
                
                # tensorboard : Train loss 기록
                writer.add_scalar(
                        tag = "Train loss",
                        scalar_value = batch_loss / batch_step,
                        global_step = epoch * len(train_dataloader)
                )

                # tensorboard : learning rate 기록
                writer.add_scalar(
                        tag = "Train LR",
                        scalar_value = optimizer.param_groups[0]['lr'],
                        global_step = epoch * len(train_dataloader)
                ) 
                
                # 배치 10개씩 처리할 때마다 lr, 평균 train loss를 출력
                if (step % 10 == 0 and step != 0):
                    learning_rate = optimizer.param_groups[0]['lr']
                    print(f"Epoch: {epoch}, Step : {step}, LR : {learning_rate}, Avg Loss : {batch_loss / batch_step:.4f}")
                    
                    # batch_loss, batch_step 초기화
                    batch_loss, batch_step = 0.0, 0

                global_step += 1
        
            if args['max_steps'] > 0 and global_step > args['max_steps']:
                break

        # 1 epoch 마다 평균 train loss 계산 및 출력
        avg_tr_loss = round(tr_loss / global_step, 4)
        avg_tr_loss_li.append(avg_tr_loss)
        print(f"Epoch {epoch} Train Loss : {avg_tr_loss}")

        # Validate!
        if valid_dataloader is not None:
            print(f"*****Epoch {epoch} Valid Start*****")
            results, val_loss = validate(args, model, device, valid_dataloader, "valid", global_step)
            avg_val_loss_li.append(val_loss)
            print(f"Epoch {epoch} Valid Loss : {val_loss:.4f}")
            print(f"*****Epoch {epoch} Train and Valid Finish*****\n")
            
            # tensorboard : Validation loss 기록 
            writer.add_scalar(
                tag = "Valid Loss",
                scalar_value = val_loss,
                global_step = epoch * len(valid_dataloader)
            )
        
        # 1 epoch 주기로 checkpoint 저장
        path = path.ckpt
        save_checkpoint(path, model, optimizer, scheduler, epoch, loss)

        mb.write("Epoch {} done".format(epoch))

        if args['max_steps'] > 0 and global_step > args['max_steps']:
            break
    
    print("Train Completed. End Program.")
    # tensorboard 기록 중지
    writer.close()  

    return global_step, tr_loss / global_step, avg_tr_loss_li, avg_val_loss_li


def validate(args, model, device, valid_dataloader, mode, global_step=None):
    results = {}

    # logger info
    if global_step != None:
        logging.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logging.info("***** Running evaluation on {} dataset *****".format(mode))
    logging.info("  Num examples = {}".format(len(v_dataset)))
    logging.info("  Eval Batch size = {}".format(args['eval_batch_size']))
		
    # loss, steps, 예측값, 라벨 초기화    
    loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label = None
    
    # validate !
    for batch in progress_bar(valid_dataloader):
        # eval mode
        model.eval()

        # inputs to device
        batch = tuple(item.to(device) for item in batch)

        # no_grad
        with torch.no_grad():
            # inputs
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]
            }
            
            # outputs
            outputs = model(**inputs)

            # loss and logits
            tmp_eval_loss, logits = outputs[:2]
            loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        
        # Get preds and out_label(예측값 & 라벨)
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label = np.append(out_label, inputs["labels"].detach().cpu().numpy(), axis=0)

    # 1 epoch 당 평균 validation loss 
    loss = loss / nb_eval_steps
    
    # preds depending on the type of task
    if seq_cls_output_modes[args['task']] == "classification":
        preds = np.argmax(preds, axis=1)
    elif seq_cls_output_modes[args['task']] == "regression":
        preds = np.squeeze(preds)

    # Pearson correlation coefficient and Spearman correlation coefficient
    result = compute_metrics("pearson_and_spearman", out_label, preds)
    results.update(result)

    return results, loss