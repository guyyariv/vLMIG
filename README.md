# Improving Visual Commonsense in Language Models via Multiple Image Generation
This repo contains the official PyTorch implementation of  [*Improving Visual Commonsense in Language Models via Multiple Image Generation*](https://pages.cs.huji.ac.il/adiyoss-lab/AudioToken/)

# Abstract
Commonsense reasoning is fundamentally based on multimodal knowledge. However, existing large language models (LLMs) are primarily trained using textual data only, limiting their ability to incorporate essential visual information. In contrast, Visual Language Models, which excel at visually-oriented tasks, often fail at non-visual tasks such as basic commonsense reasoning. 
This divergence highlights a critical challenge - the integration of robust visual understanding with foundational text-based language reasoning. To this end, we introduce a method aimed at enhancing LLMs' visual commonsense. Specifically, our method generates multiple images based on the input text prompt and integrates these into the model's decision-making process by mixing their prediction probabilities. To facilitate multimodal grounded language modeling, we employ a late-fusion layer that combines the projected visual features with the output of a pre-trained LLM conditioned on text only. This late-fusion layer enables predictions based on comprehensive image-text knowledge as well as text only when this is required. We evaluate our approach using several visual commonsense reasoning tasks together with traditional NLP tasks, including common sense reasoning and reading comprehension. Our experimental results demonstrate significant superiority over existing baselines. When applied to recent state-of-the-art LLMs (e.g., Llama3), we observe improvements not only in visual common sense but also in traditional NLP benchmarks. Code and models will be publicly available upon acceptance.

<a href="https://arxiv.org/abs/2305.13050"><img src="https://img.shields.io/badge/arXiv-2301.13826-b31b1b.svg" height=22.5></a>

# Installation
```
git clone git@github.com:guyyariv/visually_grounded_lm.git
cd visually_grounded_lm
python -m venv vlmig
source vlmig/bin/activate
pip install -r requirements.txt
```

# Pre-Trained Models
Download the pre-trained models provided in the paper using the following commands:

First, run ```pip install gdown```

#### GPT-2
```angular2html
mkdir -p output/gpt2 && \
gdown "https://drive.google.com/uc?id=1ZvJTXiuXjcCCwcm_PeQ78hxZCEab4tID" -O output/gpt2/ft_wiki_laion_220_2.bin
```

#### Gemma-2B
```angular2html
mkdir -p output/gemma_2b && \
gdown "https://drive.google.com/uc?id=14qWLXJNMcOmgMVkQa7KAdKbcucz97_o8" -O output/gemma_2b/ft_wiki_laion_220_2.bin
```

#### LLaMA 3
```angular2html
mkdir -p output/llama3 && \
gdown "https://drive.google.com/uc?id=14qWLXJNMcOmgMVkQa7KAdKbcucz97_o8" -O output/llama3/ft_wiki_laion_220_2.bin
```

# Training

Configure your Accelerate environment with:
```angular2html
accelerate config
```

Launch the training process:
```angular2html
accelerate launch train.py \
--run_name test \
--dataset_name cropped_vg \
--model_name_or_path meta-llama/Meta-Llama-3-8B \
--output_dir output/llama3 \
--report_to wandb \
--per_device_train_batch_size 64 \
--num_train_epochs 1 \
--run_bf16 \
--learning_rate 5e-4 \
--with_tracking
```
For full reproducibility, ensure to fine-tune your trained model on Wikipedia-103 (max_elements=200,000) and LAION-220.

# Evaluation

We evaluate the model on multiple benchmarks:

#### Visual Commonsense:
For ImageNetVC evaluation (based on the official implementation https://github.com/hemingkx/ImageNetVC/blob/main/VaLM/BLIP-2/ImageNetVC.py):
```angular2html
python3 eval_scripts/imagenetVC.py --model_name meta-llama/Meta-Llama-3-8B --run_name llama3_imagenetvc --pretrained_model output/llama3/ft_wiki_laion_220_2 --generate_images True --k 10 
```
Access script parameters with:
```angular2html
python3 eval_scripts/imagenetVC.py --help
```

#### Commonsense:
For CommonsenseQA evaluation:
```angular2html
python3 eval_scripts/commonsenseQA.py --model_name meta-llama/Meta-Llama-3-8B --pretrained_model output/llama3/ft_wiki_laion_220_2 --generate_images True --k 10 --testset piqa
```
Access script parameters with:
```angular2html
python3 eval_scripts/commonsenseQA.py --help
```
Note: This script can also be used to evaluate other datasets, such as SIQA, ARC, etc.

#### Reading Comprehension
For SQuAD run:
```angular2html
python3 eval_scripts/squad.py --model_name meta-llama/Meta-Llama-3-8B --pretrained_model output/llama3/ft_wiki_laion_220_2 --generate_images True --k 10
```
For QUAC and BoolQ evaluations:
```angular2html
python3 eval_scripts/quac.py
python3 eval_scripts/boolq.py
```
Access script parameters with:
```angular2html
python3 eval_scripts/squad.py --help
```

# Acknowledgments
Our code it partially built upon [transformers training example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling) and [ImagenetVC](https://github.com/hemingkx/ImageNetVC/tree/main)

# Cite
If you use our work in your research, please cite the following paper:
```
@article{yariv2024improving,
}
```

# License
This repository is released under the MIT license as found in the [LICENSE](LICENSE) file. 

