from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import random 
import numpy 
import torch.distributed as dict 
from torch import multiprocessing as mp 
import logging
import transformers 

transformers.logginng.set_verbosity_error()

logger = logging.getLogger(__name__)

def get_lm(model_name = "microsoft/DialoGPT-medium"):
    """
    Get the language model (DialoGPT) and its tokenizer
    from HuggingFace

    :param
        model_name: str
                name of a language model from HuggingFace hub

    :return
        model: PretrainedModel
                Language model (DialoGPT)
        tokenizer: PretrainedTokenizer
                Model's tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)

    return model, tokenizer

def initialize_device_settings(
        use_cuda = None, 
        local_rank = -1, 
        multi_gpu=False, 
        devices=None
        ):
    """
    Returns a list of available devices 

    :param
        use_cuda: Optional[bool]
                whther to make use of Cuda GPUs (is available)
        local_rank: int 
                Ordinal of device to be used. If -1 and `multi_gpu` is True,
                all devices will be used. Unused if `devices` is set or 
                `use_cuda` is False
        multi_gpu: bool
                Whether to make use of all GPUs (is availabel)
                Unused if `devices` os set or `use_cuda` is False 

    :return 
        devices_to_use: List[torch.device]
                a list of available devices 
        n_gpu: int 
                number of gpus 
    """
    if use_cuda is False: 
        devices_to_use = [torch.device("cpu")]
        n_gpu = 0
    elif devices: 
        if not isinstance(devices, list): 
            raise ValueError(f"devices must be a list, but got {devices} of type{type(devices)}")
        if any(isinstance(device, str) for device in devices):
            torch_devices = [torch.device(device) for device in devices]
            devices_to_use = torch_devices 
        else: 
            devices_to_use = devices 
        n_gpu = sum(1 for device in devices_to_use if "cpu" not in device.type)
    elif local_rank == -1: 
        if torch.cuda.is_available():
            if multi_gpu:
                devices_to_use = [torch.device(device) for device in range(torch.cuda.device_count())]
                n_gpu = torch.cuda.device_count()
            else: 
                devices_to_use = [torch.device("cuda")]
                n_gpu =1
        else: 
            devices_to_use = [torch.device("cpu")]
            n_gpu = 0
    else: 
        devices_to_use = [torch.device("cuda", local_rank)]
        torch.cuda.set_device(devices_to_use[0])
        n_gpu = 1
        # Initialize the distributed backend which will take care of sychronizing nodes/GPUs 
        torch.distribted.init_process_group(backend="nccl")

    logger.info(f"Using devices: {', '.join([str(device) for device in devices_to_use]).upper()}")
    logger.info(f"Number of GPUs: {n_gpu}")
    
    return devices_to_use, n_gpu
