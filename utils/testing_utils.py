from tqdm import tqdm
import torch
from utils import model_utils

from sklearn.metrics import confusion_matrix


def test(
    testing_dataloader,
    model, tokenizer,
    light_weight_tuning= None, algo = None,
    max_token_per_comment = 963,
    limit = -1,
    prefix_size = 0,
):
    if(light_weight_tuning is not None):
        assert (algo is not None), "Specify an intervention scheme {'adapter', 'prefix', 'prompt'}"
    
    print()
    print(f"testing .... ")
    print()
    
    target = []
    predict = []

    for reviews, sentiment in tqdm(testing_dataloader):
        tokenized_inputs = tokenizer(
            list(reviews),
            padding = True,
            return_tensors="pt"
        ).to(next(model.parameters()).device)

        if(tokenized_inputs['input_ids'].shape[1] > max_token_per_comment):
            # print(f"BLOCKED ==> {tokenized_inputs['input_ids'].shape[1]}")
            continue

        last_token_inds = tokenized_inputs["attention_mask"].sum(dim=1)
        max_out_len = max(last_token_inds).item()

        with torch.no_grad():
            txt, ret_dict = model_utils.generate_fast(
                model, tokenizer,
                list(reviews),
                argmax_greedy = True,
                max_out_len= prefix_size + max_out_len + 3,
                get_answer_tokens=True,
                light_weight_tuning= light_weight_tuning, algo = algo
            )

        for t, p in zip(list(sentiment), ret_dict['answer']):
            target.append(t)
            predict.append(p['top_token'])

        limit -= 1
        if(limit == 0):
            break
    
    ret_dict = {}
    try:
        tn, fp, fn, tp = confusion_matrix(target, predict).ravel()
        ret_dict['confusion_matrix'] = {
            'tp': int(tp), 'fn': int(fn),
            'fp': int(fp), 'tn': int(tn)
        }
        sensitivity = tp/(tp + fn)
        specificity = tn/(tn + fp)
        ret_dict["balanced_accuracy"] = (sensitivity + specificity)/2
    except:
        ret_dict["balanced_accuracy"] = -1

    ret_dict["target"] = target
    ret_dict["prediction"] = predict

    return ret_dict