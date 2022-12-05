import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import nethook
import re

class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
            )
            nethook.set_requires_grad(False, model)
            model.eval().cuda()
        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def print_formatted_results(prompts, txt, ret_dict):
    for i in range(len(prompts)):
        print(prompts[i])
        print(txt[i])
        if('answer' in ret_dict):
            answer = ret_dict['answer'][i]['candidates']
            print("p(answer): ", ", ".join([f"p(\'{t['token']}\'[{t['token_id']}])={t['p']}" for t in answer]))
        if('p_interesting_words' in ret_dict):
            p_interesting = ret_dict['p_interesting_words'][i]
            print("p(interesting): ", ", ".join([f"p(\'{t['token']}\'[{t['token_id']}])={t['p']}" for t in p_interesting]))

        print()
        
############################## Prompt Tuning Edit ##############################
def get_prompt_tuning_edit(soft_embeddings, embedder = "transformer.wte"):
    def insert_prompt_embeddings(output, layer, soft_embeddings = soft_embeddings):
        if(layer != embedder):
            return output
        # print("intervention ==> ", layer, "output shape ===> ", output.shape)
        # return output
        prefix_size = soft_embeddings.shape[1]
        arr = []
        for batch in output:
            added = torch.cat((soft_embeddings[0], batch[prefix_size:, :]))
            arr.append(added)
        return torch.stack(arr)
    return insert_prompt_embeddings
############################## Prompt Tuning Edit ##############################


def untuple(output):
    if(type(output) is tuple):
        return output[0]
    return output

def get_shape(output):
    pre = f"{type(output)} ==> "
    if(type(output) is tuple):
        return pre + f"{output[0].shape} -- {(output[1][0].shape, output[1][1].shape)}"
    return pre + f"{output.shape}"

############################## Prefix Tuning Edit ##############################
def get_prefix_tuning_edit(prefix_embeddings):
    def insert_prompt_embeddings(output, layer, prefix_embeddings = prefix_embeddings):
        if(layer not in prefix_embeddings):
            return output
        # print("intervention ==> ", layer, "output shape ===> ", get_shape(output))
        # return output
        X = untuple(output)
        prefix_now = prefix_embeddings[layer]
        prefix_size = prefix_now.shape[1]
        arr = []
        for batch in X:
            added = torch.cat((prefix_now[0], batch[prefix_size:, :]))
            arr.append(added)
        X = torch.stack(arr)

        if(type(output) is not tuple):
            return X
        else:
            return (X, output[1])
    return insert_prompt_embeddings
############################## Prefix Tuning Edit ##############################


############################## Adapter Tuning Edit ##############################
def get_adapter_tuning_edit(adapter_collection):
    def insert_adapters_into_calculation(output, layer, adapter_collection = adapter_collection):
        if(layer not in adapter_collection):
            return output
        # print("intervention ==> ", layer, "output shape ===> ", get_shape(output))

        return adapter_collection[layer](output)
    return insert_adapters_into_calculation
############################## Adapter Tuning Edit ##############################


import unicodedata
from typing import Optional, List
import collections
import numpy as np

def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    top_k: int = 5,
    max_out_len: int = 20,
    argmax_greedy = False,
    debug = False,

    get_answer_tokens = False,      # returns the immediate next top token and `top_k` possible candidates
    track_interesting_words = None, # for each prompt tracks the p(token) of some interesting tokens as answer (the first generated token). 
                                    # `get_answer_tokens` must be true
    light_weight_tuning = None, algo = "prompt", # options => prompt | prefix | adapter
    embedder = "transformer.wte"
):
    # print(prompts)
    tokenized = tok(prompts, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )

    intervention_function = None
    trace_modules = []

    if(light_weight_tuning is not None):
        if(algo in ["prompt", "prefix"]):
            prefix_size = light_weight_tuning.shape[1] if algo == "prompt" else light_weight_tuning[embedder].shape[1]
            # print(prefix_size, prompt_tuning.shape)
            # add soft tokens
            prefix_tokens = torch.ones(len(prompts), prefix_size, dtype = int).to(next(model.parameters()).device) * model.config.bos_token_id
            tokenized["input_ids"] = torch.cat((prefix_tokens, tokenized["input_ids"]), dim = 1)
            prefix_attn = torch.ones(len(prompts), prefix_size, dtype = int).to(next(model.parameters()).device)
            tokenized["attention_mask"] = torch.cat((prefix_attn, tokenized["attention_mask"]), dim = 1)

            if(algo == "prompt"):
                intervention_function = get_prompt_tuning_edit(light_weight_tuning, embedder)
                trace_modules = [embedder]
            else:
                intervention_function = get_prefix_tuning_edit(light_weight_tuning)
                trace_modules = list(light_weight_tuning.keys())
        else:
            intervention_function = get_adapter_tuning_edit(light_weight_tuning)
            trace_modules = list(light_weight_tuning.keys())


    # print(tokenized['input_ids'].shape)
    input_ids, attention_mask = tokenized["input_ids"], tokenized["attention_mask"]
    batch_size = input_ids.size(0)

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    # print(cur_context)

    if get_answer_tokens == True:
        prompt_lens = np.array([
            tok([p], return_tensors="pt").input_ids.shape[-1]
            for p in prompts
        ])
        if((light_weight_tuning is not None) and (algo in ["prompt", "prefix"])):
            prompt_lens += prefix_size  
        answers = [{'top_token': "<#>", 'candidates': []} for _ in range(input_ids.shape[0])]
        if(track_interesting_words is not None):
            p_interesting_words = [[] for _ in range(input_ids.shape[0])]

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            with nethook.TraceDict(
                model, trace_modules, edit_output= intervention_function
            ) as traces:
                model_out = model(
                    input_ids=input_ids[:, cur_context],
                    attention_mask=attention_mask[:, cur_context],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            if(intervention_function is not None):
                intervention_function = None
            logits, past_key_values = model_out.logits, model_out.past_key_values
            # print(" ====> ", logits.shape)

            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]

            if(argmax_greedy == False):
                new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
                new_toks = torch.gather(tk, 1, new_tok_indices)

            else:
                new_tok_indices = torch.topk(softmax_out_top_k, dim=1, k=1)
                new_toks = torch.gather(tk, 1, new_tok_indices.indices)
            
            if(get_answer_tokens == True):
                for i in range(input_ids.shape[0]):
                    if(prompt_lens[i] == cur_context.stop):
                        answers[i]['top_token'] = tok.decode(new_toks[i][0])
                        for t in tk[i]:
                            answers[i]['candidates'].append(
                                {'token': tok.decode(t), 'token_id': t.item(), 'p': round(float(softmax_out[i][int(t)]), 4)}
                            )
                        if(track_interesting_words is not None):
                            for token in track_interesting_words[i]:
                                token_id = tok(token).input_ids[0]
                                p_interesting_words[i].append(
                                    {'token': tok.decode(token_id), 'token_id': token_id, 'p': round(float(softmax_out[i][token_id]), 4)}
                                )


            if(debug == True):
                for i in range(input_ids.size(0)):
                    # print(f"{i} => ", end="")
                    token_id = new_toks[i][0]
                    print(f"\'{tok.decode([token_id])}\'[{token_id}] -- {softmax_out[i][token_id]*100}", end=" ")
                    print("[", end="")
                    for t in tk[i]:
                        # print(t)
                        print(f"\'{tok.decode(t)}\'({round(float(softmax_out[i][int(t)]*100), 3)})", end=" ")
                    print("]")

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)


    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n", " ")
        # .replace("<|endoftext|>", "")
        for x in txt
    ]

    ret_dict = {"past_key_values": past_key_values}
    if(get_answer_tokens == True):
        ret_dict['answer'] = answers
        if(track_interesting_words is not None):
            ret_dict['p_interesting_words'] = p_interesting_words
    return txt, ret_dict