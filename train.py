import argparse
import numpy as np
import torch
from transformers import DistilBertForTokenClassification,  DistilBertTokenizerFast
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from utils import get_special_tokens, trim_entity_spans, convert_goldparse, ResumeDataset, tag2idx, idx2tag, get_hyperparameters, train_and_val_model


parser = argparse.ArgumentParser(description='Train Bert-NER')
parser.add_argument('-e', type=int, default=5, help='number of epochs')
parser.add_argument('-o', type=str, default='.', help='output path to save model state')

args = parser.parse_args().__dict__

output_path = args['o']

MAX_LEN = 500
NUM_LABELS = 12
EPOCHS = args['e']
MAX_GRAD_NORM = 1.0
MODEL_NAME = 'bert-base-uncased'
# TOKENIZER = DistilBertTokenizerFast('./vocab/vocab.txt', lowercase=True)
TOKENIZER = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', lowercase=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = trim_entity_spans(convert_goldparse('data/Resumes.json'))

total = len(data)
train_data, val_data = data[:180], data[180:]

train_d = ResumeDataset(train_data, TOKENIZER, tag2idx, MAX_LEN)
val_d = ResumeDataset(val_data, TOKENIZER, tag2idx, MAX_LEN)

train_sampler = RandomSampler(train_d)
train_dl = DataLoader(train_d, sampler=train_sampler, batch_size=8)

val_dl = DataLoader(val_d, batch_size=2)

# model = DistilBertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(tag2idx))
model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=NUM_LABELS)
model.resize_token_embeddings(len(TOKENIZER)) # from here: https://github.com/huggingface/transformers/issues/1805
model.to(DEVICE)
optimizer_grouped_parameters = get_hyperparameters(model, True)
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

train_and_val_model(model, TOKENIZER, optimizer, EPOCHS, idx2tag, tag2idx, MAX_GRAD_NORM, DEVICE, train_dl, val_dl)

torch.save(
    {
        "model_state_dict": model.state_dict()
    },
    f'{output_path}/model-state.bin',
)
