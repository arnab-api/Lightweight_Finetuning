# *Light-Weight* Finetuning of Language Models

Language models are cool, they can even generate stories, poems. But, let's say we only care about one task, maybe text summarization or machine translation. The prevalent paradigm is to perform *fine-tuning* on the model for that specific task. If fine-tuned on summarization, a model would become really good at summarization however will not be much good at anythinng else as we have overwritten the parameters necessary for other natural language understading tasks.

This project is aimed to investigate a new line of concepts that introduce ***light-weight*** finetuning in the context of language models. That idea is to keep the parameters of the model itself fixed and introduce a new set of parameters that kind of steers the model to perform a specific task.

This work investigates three prominent ***light-weight*** finetuning concepts.
* **Adapter-tuning** by [Houlsby et al](https://arxiv.org/abs/1902.00751).
* **Prefix-tuning** by [Li and Liang](https://arxiv.org/abs/2101.00190.) 
* **Prompt-tuning** by [Lester et al](https://arxiv.org/abs/2104.08691)

The issue is that all the three works mentioned above works with different language models and evaluate on different tasks. The goal of this project is to implement all these three approaches on a single model for a single task and compare their efficacy.

# Instructions

Check the `Demo.ipynb` file for a quick demonstration of how 
all the tuning approaches perform in a binary sentiment
classification task.

Duplicate the `conda` environment using the following command.

`conda env create -f conda_environment.yml`

## <span style="color:red">Tuned Weights: [GDrive](https://drive.google.com/drive/folders/1kagxDJzKhS6UbKesbf7_SXFJyf1K-ITK?usp=sharing)</span>
Download the tuned weights and save them in the `Saved_weights` folder.