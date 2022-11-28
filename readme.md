# `emoGPT`
### The goal is to steer a GPT model to generate text in a specific emotion.
Will experiment with different approaches.
* Fine-tuning
* Steering GPT by finding out the MLP columns (nurons) in the `fc_proj` module that promote words of a specific emotion and increase the co-efficients of those nurons (?). **Might not work**.
* Light-weight fine-tuning
    * Prompt-tuning
    * Prefix-tuning
    * Adapter-tuning
