import torch
from torch.utils.data import DataLoader, Dataset
import re
import pandas as pd
import json
import Prefix_Tuning

import os
import sys
sys.path.append('..')
from utils import nethook
from utils import model_utils
from utils import tuning_utils
from utils import testing_utils

torch.cuda.set_device(1)

print("#### Load and preprocess data")
train_df = pd.read_csv("../Data/IMDB_50K_Reviews/train.csv")
# validation_df = pd.read_csv("../Data/IMDB_50K_Reviews/validation.csv")
test_df = pd.read_csv("../Data/IMDB_50K_Reviews/test.csv")

CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|/.*/')

def cleanhtml(raw_html):
  raw_html = raw_html.replace("\\", "")
  raw_html = raw_html.replace("&#039;", "\'")
  cleantext = re.sub(CLEANR, ' ', raw_html)
  split = cleantext.strip().split(" ")
  if(split[0].isnumeric()):
    split = split[1:]
  return " ".join([w for w in split if len(w.strip()) > 0])

class GoEmotions(Dataset):
    def __init__(self, data_frame):
        self.x = []
        self.y = []

        for index, row in data_frame.iterrows():
            self.x.append("<REVIEW>: " + cleanhtml(row["review"]) + " <SENTIMENT>")
            self.y.append(" " + row["sentiment"])
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



MODEL_NAME = "gpt2-medium"
mt = model_utils.ModelAndTokenizer(MODEL_NAME, low_cpu_mem_usage=False)
model = mt.model
tokenizer = mt.tokenizer
tokenizer.pad_token = tokenizer.eos_token
print(f"Model {MODEL_NAME} initialized")
print()

######################################################
dataset_sizes = [100, 500, 1000, 5000, 10000, 20000]
prefix_size = 2
batch_size = 2
save_path = f"../Saved_weights/EXP2/Prefix_Tuning/{MODEL_NAME}"
######################################################

test_dataset = GoEmotions(test_df)
print("test dataset size: ", len(test_dataset))
testing_dataloader = DataLoader(test_dataset, batch_size=1)

for train_size in dataset_sizes:
    training_dataset = GoEmotions(train_df[:train_size])
    print("training dataset size: ", len(training_dataset))

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size)

    print("prefix size ==> ", prefix_size)
    prefix_embeddings, tuning_logs = Prefix_Tuning.get_tuned_prefixes(
        training_dataloader,
        mt,
        prefix_size = prefix_size,
        num_epochs = 1,
        # limit = 10
    )

    os.makedirs(save_path, exist_ok = True)
    torch.save(prefix_embeddings, f"{save_path}/prefix_size__{prefix_size}____data_{train_size}.pth")

    test_results = testing_utils.test(
        testing_dataloader,
        model, tokenizer,
        light_weight_tuning = prefix_embeddings, algo = "prefix",
        prefix_size = prefix_size
        # limit = 10
    )
        
    print("Balanced Accuracy => ", test_results["balanced_accuracy"])
    with open(f"{save_path}/logs_prefix_size__{prefix_size}__data_{train_size}.json", "w") as f:
        json.dump({
            "tuninig_logs": tuning_logs,
            "test_logs": test_results
        }, f)

    print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")



