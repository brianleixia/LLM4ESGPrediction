# Training Model Code Overview

This repository contains a comprehensive suite of scripts and tools designed for pre-training PLMs (BERT, RoBERTa and DistilRoBERTa) and training financial large language models (LLMs). For pre-training PLMs, we use transformers [run_mlm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py) provided by huggingface. Our training process leverages two significant open-source projects: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for efficient fine-tuning of various LLMs, and [vLLM](https://github.com/vllm-project/vllm) for deployment and training acceleration. Additionally, we benchmark our FinLlama models using the FinGPT benchmark, which is also open-sourced and available at [AI4Finance-Foundation/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT).

## LLaMA-Factory

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) is a unified platform for efficient fine-tuning of over 100 LLMs. It provides a set of scripts that are instrumental in our training process:

- `train_llms/run_pretrain.sh`: This script is used to initiate the pretraining phase of our models, which is crucial for establishing a robust foundation in language understanding.
- `train_llms/run_sft.sh`: This script facilitates the SFT (Sequence-to-Feature) fine-tuning of our models on specific financial tasks, tailoring them to the nuances of financial text and data.

## vLLM

The [vLLM](https://github.com/vllm-project/vllm) project offers tools for deploying and accelerating large language models. It supports multi-GPU deployment, which is vital for scaling our models and enhancing their performance:

- `train_llms/evaluate/run_vllm.sh`: This script is employed for deploying our models across multiple GPUs, which not only accelerates inference but also allows for parallel processing of financial data.

## FinGPT Benchmark

Our FinLlama models are benchmarked against the FinGPT benchmark, which is an open-source initiative by [AI4Finance-Foundation](https://github.com/AI4Finance-Foundation). The benchmark, available at [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT), provides a standardized set of financial tasks and datasets to evaluate the performance of financial LLMs. This ensures that our models are not only trained effectively but also meet industry standards for financial applications.

## Training Workflow

The pre-training workflow of PLMs is structured as follows:

1. **Pretraining**: We begin with pretraining our models using the `train_bert.sh` script to pre-train BERT for example.
2. **Fine-tuning on classification task**: We then proceed to fine-tune our models using the `finetune_ddp.sh` script.

The training workflow of FinLlama is structured as follows:

1. **Pretraining**: We begin with pretraining our models using the `run_pretrain.sh` script to build a foundational understanding of financial language.
2. **SFT Fine-tuning**: We then proceed to fine-tune our models using the `run_sft.sh` script, focusing on specific financial tasks to enhance their domain-specific capabilities.
3. **Model Evaluation and Deployment**: Finally, we use the `run_vllm.sh` script for multi-GPU deployment and evaluation, ensuring that our models are ready for real-world financial applications.
