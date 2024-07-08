import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer

#from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from Bio import SeqIO
from tqdm import tqdm
from time import time

from utils.tokenizer import tok_func
from utils.datasets import Dataset

local_model_path = 'models/model_TEnoTE_25epochs'
local_model_cnn_path = 'models/TF_CNN_BERT_pool_model2'

tokenizer = BertTokenizer.from_pretrained('zhihan1996/DNA_bert_6')
model = BertForSequenceClassification.from_pretrained(local_model_path, num_labels=2)

trainer = Trainer(model)

if torch.cuda.is_available():

    device = torch.device("cuda")

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model.to(device)



test_file_path = 'testing/finalDB.fasta'


def getSeqs(file_path: str) -> list:
    seqs = []
    for seq_record in SeqIO.parse(file_path, "fasta"):
        seqs.append(str(seq_record.seq))
    return seqs[:1]


begin = time()


def processSeqs(seqs: list) -> dict:
    chunks = {512: [], 700: []}
    for seq in seqs:
        if len(seq) >= 700:
            chunks[700].append(seq)
        else:
            chunks[512].append(seq)
    
    return chunks


begin = time()
seqs = getSeqs(test_file_path)
print('Loading sequences...', len(seqs))
print('Time:', time() - begin)

print('Processing sequences...')
begin = time()
chunks = processSeqs(seqs)
print('Chunks:', len(chunks[512]), len(chunks[700]))
print('Time:', time() - begin)

def windowed_func(chunk_seq: list) -> list:
    window_size = 512
    stride = 170 # ~ 1/3 of window size
    sequences = []
    for seq in chunk_seq:
        seq_windows = []
        for i in range(0, len(seq), stride):
            start = i
            end = i + window_size

            if end > len(seq):
                end = len(seq)
            seq_windows.append(seq[start:end])
        sequences.append(seq_windows)
    return sequences

counter = 0

cut_seqs = windowed_func(chunks[700])
print('Cut seqs:', len(cut_seqs))

begin = time()
outputs = []

for s in tqdm(cut_seqs):
    X_test_tokenized = tokenizer([tok_func(x) for x in s], padding=True, truncation=True, max_length=512) # Create torch dataset
    
    test_dataset = Dataset(X_test_tokenized) # Load trained model
    output, _,_ = trainer.predict(test_dataset) # Preprocess raw predictions
    outputs.append(output)

print('predictions Time:', time() - begin)
print('Predictions:', len(outputs))
print('Predictions:', outputs)


