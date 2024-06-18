import argparse
from diffusers import AutoPipelineForText2Image
import torch
from accelerate.logging import get_logger
from nltk import sent_tokenize
from datasets import load_dataset, concatenate_datasets
from transformers import (
    MODEL_MAPPING,
    AutoTokenizer,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from operator import itemgetter

import os
import sys
sys.path.append(os.getcwd())

from models.gemma.modeling_gemma_fusion_layer import GemmaForCausalLM
from models.llama3.modeling_llama3_fusion_layer import LlamaForCausalLM
from models.gpt2.modeling_gpt2_fusion_layer import GPT2LMHeadModel
from models.opt.modeling_opt_fusion_layer import OPTForCausalLM

from typing import List, Dict, Union

check_min_version("4.36.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from transformers import CLIPModel


def calculate_clip_scores_hf(model, processor, images, prompts):
    # Process images and prompts
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)

    # Move inputs to the same device as model
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    inputs['input_ids'] = inputs['input_ids'][:, -77:]
    inputs['attention_mask'] = inputs['input_ids'][:, -77:]

    # Get image and text features from CLIP model
    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity for each corresponding image-text pair
    similarities = (image_features * text_features).sum(dim=1)  # Sum over feature dimensions
    # Clamp negative values to zero
    similarities = torch.clamp(similarities, min=0)
    return similarities


def prepare_query(testset: str, step_data: Dict[str, Union[str, List[str]]], context_columns: List[str], solution_columns: Union[List[str], str], index):
    if testset in {'arc', 'obqa', 'cqa'}:
        # Handle cases where choices are a dict with 'text'
        return ' '.join([step_data[col] for col in context_columns] + [step_data[solution_columns]['text'][index]])
    elif testset == 'winogrande':
        # Replace underscore in the sentence with the option
        return step_data[context_columns[0]].replace('_', step_data[solution_columns[index]])
    elif testset == 'hs':
        # Concatenate context with each ending
        return step_data[context_columns[0]] + ' ' + step_data[solution_columns][index]
    else:
        # Default case, concatenate context with solution
        return ' '.join([step_data[col] for col in context_columns] + [step_data[solution_columns[index]]])


def get_label(label: Union[str, int], testset: str) -> int:
    if testset in {'siqa', 'winogrande'} or (testset == 'arc' and isinstance(label, str) and label.isnumeric()):
        return int(label) - 1
    return label


def load_and_configure_dataset(testset: str):
    dataset_map = {
        'piqa': ("piqa", 'validation'),
        'siqa': ("lighteval/siqa", 'validation'),
        'hs': ("Rowan/hellaswag", 'validation'),
        'winogrande': ("winogrande", ["winogrande_xs"], ['validation']),
        'arc': ("ai2_arc", ["ARC-Easy", "ARC-Challenge"], ['test', 'test']),
        'obqa': ('allenai/openbookqa', ['main'], ['test']),
        'cqa': ("tau/commonsense_qa", 'validation'),
    }

    config = dataset_map[testset]
    if isinstance(config[1], list):
        # Load and concatenate multiple datasets
        datasets = [load_dataset(config[0], spec)[part] for spec, part in zip(config[1], config[2])]
        return concatenate_datasets(datasets)
    else:
        return load_dataset(config[0])[config[1]]


def load_testset(testset: str):
    dataset = load_and_configure_dataset(testset)

    label_col = {'piqa': 'label', 'siqa': 'label', 'winogrande': 'answer', 'arc': 'answerKey',
                 'obqa': 'answerKey', 'cqa': 'answerKey', 'hs': 'label',}[testset]
    context_cols = {
        'piqa': ['goal'], 'siqa': ['context', 'question'], 'winogrande': ['sentence'], 'arc': ['question'],
        'obqa': ['question_stem'], 'cqa': ['question'], 'hs': ['ctx'],
    }[testset]
    sol_cols = {
        'piqa': ['sol1', 'sol2'], 'siqa': ['answerA', 'answerB', 'answerC'], 'winogrande': ['option1', 'option2'],
        'arc': 'choices', 'obqa': 'choices', 'cqa': 'choices', 'hs': 'endings',
    }[testset]

    return dataset, label_col, context_cols, sol_cols


def get_num_options(dataset, step, sol_cols):
    if isinstance(sol_cols, str):
        if sol_cols == 'choices':
            num_options = len(dataset[step][sol_cols]['text'])
        else:
            num_options = len(dataset[step][sol_cols])
    else:
        num_options = len(sol_cols)
    return num_options


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if "gemma" in args.model_name:
        model = GemmaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        num_added_tokens = tokenizer.add_tokens('<image>')
        model.config.image_token_index = tokenizer('<image>')[0].ids[1]
        model.config.ignore_index = -100
        model.config.pad_token_id = 0
    if "llama" in args.model_name:
        model = LlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
    elif "gpt2" in  args.model_name:
        model = GPT2LMHeadModel.from_pretrained(args.model_name, torch_dtype=torch.float16)
    elif "opt" in args.model_name:
        model = OPTForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)

    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    if args.pretrained_model is not None:
        try:
            modules = torch.load(f'{args.pretrained_model}.bin')
            new_modules = modules.copy()
            for key in modules.keys():
                new_modules[key[7:]] = modules[key]
                del new_modules[key]

            model.load_state_dict(new_modules, strict=False)

        except:
            model.load_state_dict(torch.load(f'{args.pretrained_model}.bin'))

    if args.generate_images:
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=model.dtype,
                                                         low_cpu_mem_usage=False).to(model.device)
        # Load the model and processor from Hugging Face
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(model.device)

    dataset, label_col, context_cols, sol_cols = load_testset(args.testset)

    k = args.k

    acc = 0
    counter = 0

    model.eval()
    for step, prompt in enumerate(dataset):

        label = get_label(dataset[step][label_col], args.testset)
        num_options = get_num_options(dataset, step, sol_cols)

        our_losses = []

        for option_num in range(num_options):

            query = prepare_query(args.testset, dataset[step], context_cols, sol_cols, option_num)
            scores = None
            if args.generate_images:
                if k > 1:
                    prompts = [query] * (k - 1)
                    prompts = [sent_tokenize(prompt) for prompt in prompts]
                    prompts = [prompts[i - 1][-i % len(prompts[i - 1])] for i in range(1, k)]
                else:
                    prompts = []
                prompts.append(' '.join(query.split()[-70:]))

                images = pipe(prompts, num_inference_steps=1, guidance_scale=0.0).images
                scores = calculate_clip_scores_hf(clip, model.processor, images, prompts)

            if 'gemma' in args.model_name:
                inputs = tokenizer([query], return_tensors="pt", max_length=350, padding="max_length",
                                   truncation=True).data
            else:
                inputs = tokenizer([query], return_tensors="pt").data

            inputs['scores'] = scores
            for key in inputs:
                inputs[key] = inputs[key].to('cuda')
            inputs['labels'] = inputs['input_ids']
            if args.generate_images:
                pixel_values = model.processor(images=images, return_tensors="pt").data['pixel_values'].to(
                    'cuda').to(
                    model.dtype)
                inputs['pixel_values'] = pixel_values

            with torch.no_grad():
                outputs = model(**inputs)

            our_losses.append(torch.exp(outputs.loss).item())

        mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, True: 0, False: 1}
        if label in mapping and label not in {0, 1, 2, 3}:
            label = mapping[label]

        index, element = min(enumerate(our_losses), key=itemgetter(1))
        if index == int(label):
            acc += 1
        counter += 1

        print(f'step: {counter}, accuracy: {acc / counter}')

    print(f"final accuracy: {acc / counter}")

    file_path = f'output/results/{args.run_name}.txt'
    with open(file_path, 'a') as file:
        # Writing the configuration details in a structured format
        file.write(f"model_name_or_path: {args.model_name_or_path}\n")
        file.write(f"TestSet: {args.testset}\n")
        file.write(f"k: {args.k}\n")
        # Writing accuracy results with better readability
        file.write(f"Accuracy: {acc / counter:.2%}\n")  # Formatting to show as a percentage
        file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on commonsense datasets.")
    parser.add_argument('--model_name', type=str, help="Huggingface's pretrained model name or path")
    parser.add_argument("--generate_images", type=bool, default=False, help="whether to generate images or no")
    parser.add_argument('--k', type=int, default=10, help="how many images to generate")
    parser.add_argument('--pretrained_model', type=str, default=None, help="path to pretrained model weights")
    parser.add_argument('--testset', type=str, default=None, help="testset name, should be one of [piqa, siqa, hs, winogrande, arc, obqa, cqa]")
    args = parser.parse_args()
    main(args)
