from transformers import (
    ElectraConfig,
    ElectraTokenizer,
    ElectraForSequenceClassification
)


class STSProcessor(object):
    """Processor for the KorSTS data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return [None]


seq_cls_processors = {
    "korsts": STSProcessor
}

seq_cls_tasks_num_labels = {
    "korsts": 1
}

seq_cls_output_modes = {
    "korsts": "regression"
}

CONFIG_CLASSES = {
    "koelectra-base-v3": ElectraConfig
}

TOKENIZER_CLASSES = {
    "koelectra-base-v3": ElectraTokenizer
}

MODEL_FOR_SEQUENCE_CLASSIFICATION = {
    "koelectra-base-v3": ElectraForSequenceClassification
}