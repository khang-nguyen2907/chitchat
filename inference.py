from transformers import AutoModelWithLMHead, AutoTokenizer 
import torch
import logging 
from utils import (
        get_lm,
        initialize_device_settings
        )
import transformers 

transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)

def _reprocess_history(
        context, 
        thresh = 4, 
        eos_token_id=50256, 
        reset = False): 
    """
    check if history exceed threshold. 

    :param 
        context: torch.tensor
                input tensor 
        thresh: int 
                if number of context exceed it or not 
        eos_token_id: int 
                eos id used to count the number of context, 
                because each context ends with eos_token_id
        reset: bool 
                empty the history context or just cut it off 

    :return 
        context: torch.tensor
                an empty tensor or cut off tensor 
    """
    pos_eos = (context[0] ==eos_token_id).nonzero(as_tuple=True)[0]
    num_context = pos_eos.size(0)
    if num_context  > thresh: 
        if reset: 
            context = torch.tensor([[]])
        else: 
            pos = int(pos_eos[-thresh-1])
            context = context[:, pos+1: ]
    return context


def run_chitchat(
        model, 
        tokenizer, 
        devices, 
        max_length = 200, 
        top_k = 2, 
        top_p = 0.9, 
        temperature = 1.0, 
        repetition_penalty = 1.1, 
        reset = False, 
        thresh_reset = 4
        ):
    """
    Start a conversation with a chitchat bot.

    :param
        model: PretrainedModel 
                a language model for text generation 
        tokenizer: PretrainedTokenizer
                a tokenizer for creating inputs for the `model`
        devices: List[torch.device]
                specify to model to use cpu or gpu(s)
        max_length: int 
                an allowed length of the input put into `model` 
                including context and new generated text 
        top_k: int 
                top_k potential candidates for the next word prediction 
                done by model generation task 
        top_p: float
                choose candidates with the probability is higher than top_p 
        temperature: float 
                Sharp the probability 
        repetition_penalty: float 
                avoid repeated generated output compared to existing context 
        reset: bool 
                whether remove all context in history or just cut some context off 
        thresh_reset: int 
                How many contexts are kept after the history is cut off 
                or after how many context in history then the history is removed all 

    :return 
        None 

    """
    step = 0
    logger.info("Chatbot is starting...")
    logger.info("Please type `:q` to stop the conservation")
    while True: 
        user = str(input(">> User: "))
        if user.startswith(":q"): 
            print("Bye! See you soon.")
            break

        new_input_ids = tokenizer.encode(user+tokenizer.eos_token, return_tensors="pt").to(devices)
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim = -1) if step > 0 else new_input_ids
        chat_history_ids = model.generate(
                bot_input_ids, 
                max_length = max_length, 
                pad_token_id = tokenizer.eos_token_id, 
                do_sample = True, 
                top_k = top_k, 
                top_p = top_p, 
                temperature = temperature, 
                no_repeat_ngram_size = 3, 
                repetition_penalty = repetition_penalty
                )
        print(">> Bot: ", tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens = True))
        step += 1
        
        chat_history_ids = _reprocess_history(
                chat_history_ids,
                thresh=thresh_reset, 
                eos_token_id = tokenizer.eos_token_id, 
                reset = reset
                ) 
            

if __name__ == "__main__": 
    model, tokenizer = get_lm()
    devices, _ = initialize_device_settings(use_cuda=True)
    model.to(device)
    model.eval()


    run_chitchat(
            model,
            tokenizer, 
            devices, 
            max_length = 200, 
            top_k = 2, 
            top_p = 0.9, 
            temperature = 1.0, 
            repetition_penalty = 1.1, 
            reset = False
            )
