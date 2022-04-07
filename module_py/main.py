import os, sys
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.processor import (
    STSProcessor,
    seq_cls_output_modes, seq_cls_processors, seq_cls_tasks_num_labels,
    CONFIG_CLASSES, MODEL_FOR_SEQUENCE_CLASSIFICATION, 
)
from src.utils import args, path, init_logger, set_seed
from data.make_inputs import train_dataloader, valid_dataloader
from modeling.train import train, validate

def main(args):
    init_logger()
    set_seed(args)
    
    # STSProcessor
    processor = seq_cls_processors[args['task']](args)
    labels = processor.get_labels()
    
    # regression 방식으로 train 
    if seq_cls_output_modes[args['task']] == "regression":
        config = CONFIG_CLASSES[args['model_type']].from_pretrained(
            args['model_name_or_path'],
            num_labels=seq_cls_tasks_num_labels[args['task']]
        )
    else:
        raise KeyError("The task is not 'regression' !")
    
    # 모델 (koelectra-base-v3-discriminator)
    model = MODEL_FOR_SEQUENCE_CLASSIFICATION[args['model_type']].from_pretrained(
        args['model_name_or_path'],
        config=config
    )

    # GPU or CPU
    # args['device'] = "cuda" if torch.cuda.is_available() and not args['no_cuda'] else "cpu"
    # model.to(args['device'])
    model.to(device)
    
    # Train !
    if args['do_train']:
        global_step, tr_loss, avg_tr_loss_li, avg_val_loss_li = train(args, model, device, writer, train_dataloader, valid_dataloader)
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))
    else:
        raise AssertionError(args['do_train'])

    return  global_step, tr_loss, avg_tr_loss_li, avg_val_loss_li

# driver
if __name__ == "__main__":
    # device type
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"# available GPUs : {torch.cuda.device_count()}")
        print(f"GPU name : {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
    print(device)

    # log 파일을 저장할 경로를 지정
    logdir_path = path['logdir']
    writer = SummaryWriter(logdir_path)

    # '__main__' 이름의 logger 생성
    logger = logging.getLogger(__name__)

    # Go !
    global_step, tr_loss, avg_tr_loss_li, avg_val_loss_li = main(args)