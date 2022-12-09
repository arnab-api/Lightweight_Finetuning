import torch
from torch.nn import Linear, ReLU, Sequential


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


def get_adapter_tuning_edit(adapter_collection): # to be used with the `nethook` module
    def insert_adapters_into_calculation(output, layer, adapter_collection = adapter_collection):
        if(layer not in adapter_collection):
            return output
        # print("intervention ==> ", layer, "output shape ===> ", get_shape(output))

        return adapter_collection[layer](output)
    return insert_adapters_into_calculation
###################################### Adapter ######################################
