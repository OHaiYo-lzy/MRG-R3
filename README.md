# MRG-R3
Code for MRG-R3: Retrieval-Reflection-Reward via Multimodal LLMs Advances Clinical Automated Medical Report Generation

## Overview

## System Requirements
### Hardware requirements
### Software requirements
### Python Dependencies


##  Installation Guide
To install using pip:
```shell
pip install ms-swift -U
```

To install from source:
```shell
# pip install git+https://github.com/modelscope/ms-swift.git
# git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

Running Environment:

|              | Range        | Recommended | Notes                                     |
| ------------ |--------------| ----------- | ----------------------------------------- |
| python       | >=3.9        | 3.10        |                                           |
| cuda         |              | cuda12      | No need to install if using CPU, NPU, MPS |
| torch        | >=2.0        |             |                                           |
| transformers | >=4.33       | 4.51      |                                           |
| modelscope   | >=1.23       |             |                                           |
| peft | >=0.11,<0.16 | ||
| trl | >=0.13,<0.18 | 0.17 |RLHF|
| deepspeed    | >=0.14       | 0.14.5 | Training                                  |
| vllm         | >=0.5.1      | 0.7.3/0.8       | Inference/Deployment/Evaluation           |
| lmdeploy     | >=0.5        | 0.8       | Inference/Deployment/Evaluation           |
| evalscope | >=0.11       |  | Evaluation |


## Quick Start

### Command Line Interface
- Here, `--adapters` should be replaced with the checkpoint folder. 
- Since the adapters folder contains the training parameter file `args.json`, there is no need to specify `--model`, `--system` separately; Swift will automatically read these parameters. To disable this behavior, you can set `--load_args false`.

```shell
# Using an interactive command line for inference.
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --temperature 0 \
    --max_new_tokens 2048

# merge-lora and use vLLM for inference acceleration
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --merge_lora true \
    --infer_backend vllm \
    --max_model_len 8192 \
    --temperature 0 \
    --max_new_tokens 2048
```

## 🏛 License

This framework is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE). For models and datasets, please refer to the original resource page and follow the corresponding License.
