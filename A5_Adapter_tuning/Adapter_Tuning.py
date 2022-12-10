import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import os
import sys
sys.path.append('..')
from utils import nethook
from utils import model_utils
from utils.tuning_utils import Adapter, get_initial_set_of_adapters, get_adapter_tuning_edit


def get_tuned_adapters(
    training_dataloader, # torch.Dataloader
    mt = "gpt2-medium",
    ## adapter specific params
    adapter_dim = 128,
    hidden_conf = [],
    ## tune specific params
    num_epochs = 10,
    learning_rate = 5e-4,
    warmup_steps = 200,
    weight_decay = 0,
    max_token_per_comment = 963
):
    if(type(mt) == str):
        MODEL_NAME = model
        mt = model_utils.ModelAndTokenizer(MODEL_NAME, low_cpu_mem_usage=False)
        model = mt.model
        tokenizer = mt.tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Model {MODEL_NAME} initialized")
    else:
        model, tokenizer = mt.model, mt.tokenizer

    adapter_blocks = get_initial_set_of_adapters(
        model, adapter_dim = adapter_dim, hidden_conf = hidden_conf
    )
    insert_adapter_embeddings = get_adapter_tuning_edit(adapter_blocks)
    print(f"Initializing adapters with adapter_size = {adapter_dim} and hidden_conf = {hidden_conf}")

    for name, w in model.named_parameters():
        w.requires_grad = True

    tunable = []
    for k in adapter_blocks:
        tunable += adapter_blocks[k].parameters()

    for t in tunable:
        t.requires_grad = True

    optimizer = torch.optim.Adam(
        tunable,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    training_loss_track = []

    print("tuning ..... ")
    print()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        for reviews, sentiments in tqdm(training_dataloader):
            tokenized_inputs = tokenizer(
                list(reviews),
                padding = True,
                return_tensors="pt"
            ).to(next(model.parameters()).device)

            if(tokenized_inputs['input_ids'].shape[1] > max_token_per_comment):
                # print(f"BLOCKED ==> {tokenized_inputs['input_ids'].shape[1]}")
                continue

            target_ids = tokenizer(
                list(sentiments), 
                padding = True,
                return_tensors="pt"
            ).to(next(model.parameters()).device)['input_ids']

            last_token_inds = tokenized_inputs["attention_mask"].sum(dim=1) - 1
            loss_mask = target_ids != tokenizer.unk_token_id

            # tokenized["input_ids"].require_grad = True
            with nethook.TraceDict(
                model,
                list(adapter_blocks.keys()),
                edit_output=insert_adapter_embeddings
            ) as traces:
                outputs = model(
                    **tokenized_inputs, 
                    labels=tokenized_inputs['input_ids']
                )

            batch_size = len(sentiments)
            probs = torch.nn.functional.log_softmax(
                outputs.logits[torch.arange(batch_size), last_token_inds], dim=-1
            )
            loss = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(1) / loss_mask.sum(1)
            loss = loss.mean()
            training_loss_track.append(loss.item())

            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    ret_dict = {}
    ret_dict["training_loss_track"] = training_loss_track
    return adapter_blocks, ret_dict
