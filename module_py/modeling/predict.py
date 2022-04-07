import os, sys
import numpy as np
import logging
import torch
from fastprogress.fastprogress import progress_bar

# 절대 경로 참조
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data.make_inputs import d_dataset
from src.processor import seq_cls_output_modes
# from main import device


def predict(args, model, device, dev_dataloader, mode, global_step=None):
    
    # logging info
    if global_step != None:
        logging.info("***** Running test on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logging.info("***** Running test on {} dataset *****".format(mode))
    logging.info("  Num examples = {}".format(len(d_dataset)))
    logging.info("  Eval Batch size = {}".format(args['eval_batch_size']))
    
    # loss, steps, preds, labels
    loss = 0.0
    nb_test_steps = 0
    preds = None
    out_label_ids = None
    
    # eval mode
    model.eval()

    # model to device
    model.to(device)

    # Predict !
    for batch in progress_bar(dev_dataloader):
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
            tmp_test_loss, logits = outputs[:2]
            loss += tmp_test_loss.mean().item()
        
        nb_test_steps += 1

        # preds and labels
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    
    # average loss
    loss = loss / nb_test_steps
    
    # preds
    if seq_cls_output_modes[args['task']] == "classification":
        preds = np.argmax(preds, axis=1)
    elif seq_cls_output_modes[args['task']] == "regression":
        preds = np.squeeze(preds)

    """ returns info
    preds           :  predictions(float)
    out_label_ids   :  labels(float)
    loss            :  loss of prediction
    """
    return preds, out_label_ids, loss