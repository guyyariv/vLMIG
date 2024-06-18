import argparse
import torch
from diffusers import AutoPipelineForText2Image
from nltk import sent_tokenize
from transformers import LlamaForCausalLM, AutoTokenizer, \
    OPTForCausalLM, CLIPModel

import os
import sys
sys.path.append(os.getcwd())

from models.gemma.modeling_gemma_fusion_layer import GemmaForCausalLM
from models.llama3.modeling_llama3_fusion_layer import LlamaForCausalLM
from models.gpt2.modeling_gpt2_fusion_layer import GPT2LMHeadModel
from models.opt.modeling_opt_fusion_layer import OPTForCausalLM

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import softmax


def preprocess_function(examples):
    return {'input_texts': f"Context: {examples['passage'].capitalize()}. Question: {examples['question'].capitalize()}? Answer:",
            'labels': examples['answer']}


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


def generate_predictions(model, tokenizer, text, device, yes_token_id, no_token_id, pipe=None, clip=None, k=1, model_name='gemma'):
    if model_name == 'gemma':
        inputs = tokenizer([text], return_tensors="pt", max_length=1500, padding="max_length",
                           truncation=True).to(device)
    elif model_name != 'blip' and model_name != 'llava':
        inputs = tokenizer(text, return_tensors="pt").to(device)

    if args.generate_images:
        if k > 1:
            prompts = [text] * (k - 1)
            prompts = [sent_tokenize(prompt) for prompt in prompts]
            prompts = [prompts[i - 1][-i % len(prompts[i - 1])] for i in range(1, k)]
        else:
            prompts = []
        prompts.append(' '.join(text.split()[-60:]))
        images = pipe(prompts, num_inference_steps=1, guidance_scale=0.0).images

        if model_name == 'gemma' or model_name == 'gpt2' or model_name == 'llama':
            # scores = calculate_clip_scores_hf(clip, model.processor, images, prompts)
            # inputs['scores'] = scores
            pixel_values = model.processor(images=images, return_tensors="pt").data['pixel_values'].to(
                'cuda').to(
                model.dtype)
            inputs['pixel_values'] = pixel_values

    if 'blip' in model_name:
        inputs = tokenizer(images=images, text=prompts, return_tensors="pt").to("cuda", torch.float16)
        inputs['labels'] = inputs['input_ids']

    if 'llava' in model_name:
        query = "<image>\n" + prompts[0]
        inputs = tokenizer(images=images, text=query, return_tensors="pt").to("cuda", torch.float16)
        inputs['labels'] = inputs['input_ids']

    with torch.no_grad():
        output = model(**inputs)
    logits = output.logits[:, -1, :] # Get logits of the last token

    # Mask all tokens except 'yes' and 'no'
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[:, yes_token_id] = True
    mask[:, no_token_id] = True
    masked_logits = torch.where(mask, logits, torch.tensor(float('-inf')).to(device))

    probs = softmax(masked_logits, dim=-1)
    predicted_token_id = probs.argmax().item()
    prediction = tokenizer.decode(predicted_token_id)
    return prediction.strip().lower()


def compute_accuracy(predictions, labels):
    correct = sum([1 if pred == label else 0 for pred, label in zip(predictions, labels)])
    accuracy = correct / len(predictions)
    return accuracy


def main(args):
    dataset = load_dataset('google/boolq')
    processed_data = dataset.map(preprocess_function)
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
    device = model.device

    if args.generate_images:
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=model.dtype,
                                                         low_cpu_mem_usage=False).to(model.device)
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(model.device)

    else:
        pipe = None
        clip = None

    if args.pretrained_model is not None:
        modules = torch.load(f"{args.pretrained_model}.bin")
        new_modules = modules.copy()
        for key in modules.keys():
            new_modules[key[7:]] = modules[key]
            del new_modules[key]
        model.load_state_dict(new_modules)

    yes_token_id = tokenizer.encode(' Yes', add_special_tokens=False)[0]
    no_token_id = tokenizer.encode(' No', add_special_tokens=False)[0]

    test_loader = DataLoader(processed_data['validation'], batch_size=1)
    if args.model_name == 'gpt2':
        test_loader = DataLoader(processed_data['validation'].select(range(3144)), batch_size=1)

    model.eval()
    predictions = []
    map_label = {'True': 'yes', 'False': 'no'}
    counter = 0
    labels = [map_label[str(label)] for label in processed_data['validation']['answer']]
    with torch.no_grad():
        for batch in tqdm(test_loader):
            for text in batch['input_texts']:
                pred = generate_predictions(model, tokenizer, text, device, yes_token_id, no_token_id, pipe, clip, args.k, args.model_name)
                predictions.append(pred)
                counter += 1

    accuracy = compute_accuracy(predictions, labels)
    print(f"Accuracy: {accuracy:.3f}")
    print(counter)

    file_path = f'output/results/{args.model_name}.txt'
    with open(file_path, 'a') as file:
        # Writing the configuration details in a structured format
        file.write(f"model_name_or_path: {args.model_name}\n")
        file.write(f"TestSet: boolq\n")
        file.write(f"k: {args.k}\n")
        file.write(f"Accuracy: {accuracy:.3f}")  # Formatting to show as a percentage
        file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on BoolQ dataset.")
    parser.add_argument('--model_name', type=str, help="Huggingface's pretrained model name or path")
    parser.add_argument("--generate_images", type=bool, default=False, help="whether to generate images or no")
    parser.add_argument('--k', type=int, default=10, help="how many images to generate")
    parser.add_argument('--pretrained_model', type=str, default=None, help="path to pretrained model weights")
    args = parser.parse_args()
    main(args)
