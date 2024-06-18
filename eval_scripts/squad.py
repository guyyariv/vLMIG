import json
import re
import string
from collections import Counter
import argparse

from nltk import sent_tokenize
from transformers import AutoTokenizer
from diffusers import AutoPipelineForText2Image
import torch
from datasets import load_dataset

import os
import sys
sys.path.append(os.getcwd())

from models.gemma.modeling_gemma_fusion_layer import GemmaForCausalLM
from models.llama3.modeling_llama3_fusion_layer import LlamaForCausalLM
from models.gpt2.modeling_gpt2_fusion_layer import GPT2LMHeadModel
from models.opt.modeling_opt_fusion_layer import OPTForCausalLM
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


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(model_name, model, tokenizer, dataset, max_length=512, generate_images=False, k=1):
    if generate_images:
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=model.dtype,
                                                         low_cpu_mem_usage=False).to(model.device)
        # Load the model and processor from Hugging Face
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(model.device)

    former_title = ''
    skip_flag = False
    f1 = exact_match = total = 0
    for item in dataset:
        title = item['title']
        if not item['answers']['text']:
            continue
        if former_title == title:
            former_question = question
            former_answer = answers[0]
        else:
            skip_flag = True
        former_title = title
        context = item['context']
        question = item['question']
        answers = item['answers']['text']
        if skip_flag:
            skip_flag = False
            continue
        skip_flag = False

        # input_prompt = context + " Question: " + question + " Answer:"
        input_prompt = f'Answer each question using information in the preceding background paragraph. ' \
                       f'If there is not enough information provided, answer with "Not in background."\n\n' \
                       f'Title: {title}\n\n' \
                       f'Background: {context}\n\n' \
                       f'Q: {former_question}\n\n' \
                       f'A: {former_answer}\n\n' \
                       f'Q: {question}\n\n' \
                       f'A:'

        if 'gemma' in model_name:
            input_text = tokenizer.encode(input_prompt,
                                          return_tensors="pt", max_length=750, padding="max_length",
                                          truncation=True).to("cuda")
        elif model_name != 'blip':
            input_text = tokenizer.encode(input_prompt,
                                          return_tensors="pt").to("cuda")
        max_new_tokens = 0
        for answer in answers:
            max_new_tokens = max(max_new_tokens, len(answer.split()))
        if generate_images:

            if k > 1:
                prompts = [input_prompt.replace(' Answer:', '')] * (k-1)
                prompts = [sent_tokenize(prompt) for prompt in prompts]
                prompts = [prompts[i-1][-i % len(prompts[i-1])] for i in range(1, k)]
            else:
                prompts = []
            prompts.append(' '.join(input_prompt.split()[-70:]))

            images = pipe(prompts, num_inference_steps=1, guidance_scale=0.0).images

            scores = calculate_clip_scores_hf(clip, model.processor, images, prompts)
            pixel_values = model.processor(images=images, return_tensors="pt").data['pixel_values'].to('cuda').to(
                model.dtype)
            output_text_ids = model.generate(input_text, pixel_values=pixel_values, max_new_tokens=50,
                                             scores=scores)
            prediction = \
            tokenizer.decode(output_text_ids[0], skip_special_tokens=True)[len(input_prompt):].strip().split('\n')[
                0]

        else:
            output_text_ids = model.generate(**inputs, max_new_tokens=50)
            prediction = \
                tokenizer.decode(output_text_ids[0], skip_special_tokens=True)[len(input_prompt):].strip().split('\n')[0]

        print(prediction)
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, answers)
        f1 += metric_max_over_ground_truths(f1_score, prediction, answers)
        total += 1
        print('total:', total)
        print('exact_match:', exact_match / total)
        print('f1:', f1 / total, '\n')
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


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
        modules = torch.load(f"{args.pretrained_model}.bin")
        new_modules = modules.copy()
        for key in modules.keys():
            new_modules[key[7:]] = modules[key]
            del new_modules[key]
        model.load_state_dict(new_modules)
    dataset = load_dataset("rajpurkar/squad_v2")['validation']
    results = evaluate(args.model_name, model, tokenizer, dataset, generate_images=args.generate_images, k=args.k)
    print(json.dumps(results))
    file_path = f'output/results/{args.model_name}.txt'
    with open(file_path, 'a') as file:
        # Writing the configuration details in a structured format
        file.write(f"model_name_or_path: {args.model_name}\n")
        file.write(f"TestSet: SQuAD\n")
        file.write(f"k: {args.k}\n")
        file.write(f"exact_mach: {results['exact_match']:.3f}")  # Formatting to show as a percentage
        file.write(f"f1: {results['f1']:.3f}")  # Formatting to show as a percentage
        file.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    parser.add_argument("--generate_images", type=bool, default=False, help="whether to generate images or no")
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to the pretrained model')
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B", help='Model name')
    parser.add_argument('--k', type=int, default=5, help='num of generated images')
    args = parser.parse_args()
    main(args)
