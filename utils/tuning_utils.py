import torch
from torch.nn import Linear, ReLU, Sequential
from utils import nethook

###################################### MISC ######################################
def untuple(output):
    if(type(output) is tuple):
        return output[0]
    return output

def get_shape(output):
    pre = f"{type(output)} ==> "
    if(type(output) is tuple):
        return pre + f"{output[0].shape} -- {(output[1][0].shape, output[1][1].shape)}"
    return pre + f"{output.shape}"
###################################### MISC ######################################


###################################### Adapter ######################################
def init_weights(m, lo = -0.0001, hi = 0.0001):
    if isinstance(m, Linear):
        torch.nn.init.uniform_(m.weight, a = lo, b = hi)
        torch.nn.init.uniform_(m.bias, a = lo, b = hi)


class Adapter(torch.nn.Module):
    def __init__(self, inp_out_dim, adapter_dim, hidden_conf = []):
        super().__init__()
        self.inp_out_dim = inp_out_dim
        self.conf = [inp_out_dim] + hidden_conf + [adapter_dim] + hidden_conf[::-1] + [inp_out_dim]
        # print(self.conf)
        self.adapter_dim = adapter_dim
        self.layers = []

        i = 1
        while i < len(self.conf):
            inp = self.conf[i-1]
            out = self.conf[i]
            layer_name = f'layer{i}'
            setattr(self, layer_name, Sequential(Linear(inp, out), ReLU()))
            self.layers.append(layer_name)
            i += 1
    
    def forward(self, x):
        x_init = x.clone()
        for module in self.named_children():
            layer_name = module[0]
            layer = getattr(self, layer_name)
            x = layer(x)
        return x + x_init


def get_initial_set_of_adapters(
    model, 
    adapter_dim = 128,
    hidden_conf = [], # no hidden layers
    initialize_as_identity = True, # if set to true, the weights of the adapters will be initilized with close to zero. to keep the contribution as identity (most likely) 
):
    mlp_blocks = [f"transformer.h.{n}.mlp" for n in range(model.config.n_layer)]
    adapter_blocks = {
        k: Adapter(
            inp_out_dim = model.config.n_embd,
            adapter_dim = adapter_dim,
            hidden_conf = hidden_conf
        ).to(next(model.parameters()).device)
        for k in mlp_blocks
    }

    if(initialize_as_identity):
        for k in adapter_blocks:
            adapter_blocks[k] = adapter_blocks[k].apply(init_weights)
    return adapter_blocks


def get_adapter_tuning_edit(adapter_collection): # to be used with the `nethook` module
    def insert_adapters_into_calculation(output, layer, adapter_collection = adapter_collection):
        if(layer not in adapter_collection):
            return output
        # print("intervention ==> ", layer, "output shape ===> ", get_shape(output))

        return adapter_collection[layer](output)
    return insert_adapters_into_calculation
###################################### Adapter ######################################


###################################### Prefix ######################################
import random
#######################################################
init_words = ["sentiment", "elephant", "review"]
#######################################################

def get_initial_prefix(
    model, tokenizer,
    embedder = "transformer.wte",
    prefix_size = 5
):
    words = random.choices(init_words, k=prefix_size)
    sentence = " " + " ".join(words)
    tokenized = tokenizer(sentence, return_tensors = "pt").to(next(model.parameters()).device)
    embedder_module = nethook.get_module(model, embedder)
    return embedder_module(tokenized['input_ids'])


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
###################################### Prefix ######################################

###################################### Prompt ######################################
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
###################################### Prompt ######################################
