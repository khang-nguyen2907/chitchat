# DialoGPT chatchit 

This is a chatchit bot using pretrained DialoGPT [[paper](https://arxiv.org/abs/1911.00536), [project](https://github.com/microsoft/DialoGPT)] model 
## Installation
```bash
$ conda create -n chatchit python=3.7 -y 
$ conda activate chatchit
$ pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install numpy transformers 
$ cd chitchat
```

## Usage
To start the chatchit bot
```python
python inference.py
```
- Feel free to access the code in `inference.py` for change generation hyper-parameters such as `top_k`, `top_p`, `temperature`, `repetition_penalty` and so on 
- If we keep the history (or context) from previous utterances during the conversation to generate the next bot's respond. It is long and noisy, later generation results get worse. Therefore, I suggest two ways dealing with it: 
    - `reset=True`: when the history storage has `thresh_reset` number of context (or previous utterances), the history will be cleaned and become empty 
    - `reset=False`: when the number of contexts (or previous utterances) exceed `thresh_reset`, the history storage will be cut off and always committed to only have `thresh_reset` number of utterances kept in the history
- Pretrained model can be experimented with small version or large version. Here I test it on medium version. Take a look at [here](https://github.com/microsoft/DialoGPT) for your pretrained model choice
## Future work
- This is just an experiment version with only using pretrained model, there is no any fine-tuning step done on it. In order to improve performance, this chatchit bot needs fine-tuning on other datasets to make it smoother. 
- Modify the usage of history, look for the solution to reduce its bad effects on generation results when becoming too long.
