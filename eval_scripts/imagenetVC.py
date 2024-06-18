## based on https://github.com/hemingkx/ImageNetVC/blob/main/VaLM/BLIP-2/ImageNetVC.py

import os
import json
import pandas
import numpy as np
from tqdm import tqdm
import argparse
import torch
from transformers import AutoConfig, AutoTokenizer, CLIPModel
from diffusers import AutoPipelineForText2Image

import sys
sys.path.append(os.getcwd())

from models.gpt2.modeling_gpt2_fusion_layer import GPT2LMHeadModel
from models.gemma.modeling_gemma_fusion_layer import GemmaForCausalLM
from models.llama3.modeling_llama3_fusion_layer import LlamaForCausalLM
from models.opt.modeling_opt_fusion_layer import OPTForCausalLM


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


def write_json(results, subset='color', run_name="test", path='output/results', prompt_idx=0):
    type_path = os.path.join(path, run_name)
    if not os.path.exists(type_path):
        os.mkdir(type_path)
    subset_path = os.path.join(type_path, subset)
    if not os.path.exists(subset_path):
        os.mkdir(subset_path)
    save_path = os.path.join(subset_path, "prompt{}".format(prompt_idx) + '.json')
    with open(save_path, "w") as f:
        json.dump(results, f)


def load_candidates(subset='color'):
    if subset == 'color':
        candidates = ['brown', 'black', 'white', 'yellow', 'green', 'gray', 'red', 'orange', 'blue', 'silver', 'pink']
    elif subset == 'shape':
        candidates = ['round', 'rectangle', 'triangle', 'square', 'oval', 'curved', 'cylinder', 'straight',
                      'cone', 'curly', 'heart', 'star']
    elif subset == 'material':
        candidates = ['metal', 'wood', 'plastic', 'cotton', 'glass', 'fabric', 'stone', 'rubber', 'ceramic',
                      'cloth', 'leather', 'flour', 'paper', 'clay', 'wax', 'concrete']
    elif subset == 'component':
        candidates = ['yes', 'no']
    elif subset == 'others_yes':
        candidates = ['yes', 'no']
    elif subset == 'others_number':
        candidates = ['2', '4', '6', '1', '8', '3', '5']
    elif subset == 'others_other':
        candidates = ['long', 'small', 'short', 'large', 'forest', 'water', 'ocean', 'big', 'tree', 'ground', 'tall',
                      'wild', 'outside', 'thin', 'head', 'thick', 'circle', 'brown', 'soft', 'land', 'neck', 'rough',
                      'chest', 'smooth', 'fur', 'hard', 'top', 'plants', 'black', 'metal', 'books', 'vertical', 'lake',
                      'grass', 'road', 'sky', 'front', 'kitchen', 'feathers', 'stripes', 'baby', 'hair', 'feet',
                      'mouth', 'female', 'table']
    else:
        print("Subset does not exist!")
        candidates = []
    return candidates


def load_prompt(question, idx=0):
    prompts = ["{}".format(question),
               "{} Answer:".format(question),
               "{} The answer is".format(question),
               "Question: {} Answer:".format(question),
               "Question: {} The answer is".format(question),
               ]
    return prompts[idx]


def load_demonstrations(subset, idx=0):
    df = pandas.read_csv('evaluation_data/imagenetvc/dev/{}.csv'.format(subset), header=0)
    demonstrations = ""
    for i in range(len(df)):
        question = df['question'][i]
        answer = df['answer'][i]
        prompts = [
                   "{} {}. ".format(question, answer),
                   "{} Answer: {}. ".format(question, answer),
                   "{} The answer is {}. ".format(question, answer),
                   "Question: {} Answer: {}. ".format(question, answer),
                   "Question: {} The answer is {}. ".format(question, answer),
                   ]
        demonstrations += prompts[idx]
    return demonstrations


def _test(model, vis_processors, subset='color', run_name='test', prompt_idx=0, icl=False,
          generate_images=False, k=10):
    if generate_images:
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=model.dtype,
                                                         low_cpu_mem_usage=False).to(model.device)
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(model.device)
    else:
        pipe = None
        clip = None

    model.eval()
    cnt = 0
    correct = 0
    df = pandas.read_csv('evaluation_data/imagenetvc/{}.csv'.format(subset), header=0)
    results = []
    for i in tqdm(range(len(df))):
        cnt += 1
        sub_subset = None
        question = df['question'][i]
        answer = str(df['answer'][i]).lower()
        if subset == 'others':
            if answer in ['yes', 'no']:
                sub_subset = 'others_yes'
            elif answer in ['2', '4', '6', '1', '8', '3', '5']:
                sub_subset = 'others_number'
            else:
                sub_subset = 'others_other'
            str_candidates = load_candidates(sub_subset)
        else:
            str_candidates = load_candidates(subset)

        try:
            token_idx = 1 if vis_processors.add_bos_token else 0
        except:
            token_idx = 1
        candidates = [vis_processors.encode(' ' + cand)[token_idx] for cand in str_candidates]
        candidates += [vis_processors.encode(' ' + cand.capitalize())[token_idx] for cand in str_candidates]

        prefix = load_prompt(question, prompt_idx)
        prompt = prefix

        if icl:
            if subset == 'others':
                demonstrations = load_demonstrations(sub_subset, idx=prompt_idx)
            else:
                demonstrations = load_demonstrations(subset, idx=prompt_idx)
            prefix = demonstrations + prefix

        # inputs = vis_processors(text=[prefix], return_tensors="pt", max_length=150, padding="max_length", truncation=True).to(model.device).data
        inputs = vis_processors(text=[prefix], return_tensors="pt").to(model.device).data

        if generate_images:

            images = pipe([prompt]*k, num_inference_steps=1, guidance_scale=0.0).images
            scores = calculate_clip_scores_hf(clip, model.processor, images, [prompt]*k)
            inputs['scores'] = scores * 2

            pixel_values = model.processor(images=images, return_tensors="pt").data['pixel_values'].to(
                'cuda').to(
                model.dtype)
            inputs['pixel_values'] = pixel_values

        outputs = model(**inputs)
        output_logits = outputs.logits[:, -1, :]
        logits = output_logits.softmax(dim=-1).cpu().mean(dim=0)

        predicted_color = {vis_processors.decode([color_index]): logits[color_index].item() for color_index in
                           candidates}
        predicted_color = sorted(predicted_color.items(), key=lambda x: x[1], reverse=True)

        result_line = {"question_id": cnt, "answer": predicted_color[0][0].strip()}
        results.append(result_line)

        if predicted_color[0][0].strip().lower() == answer.strip():
            correct += 1

        print(correct/cnt)
    write_json(results, subset, run_name, prompt_idx=prompt_idx)
    print("Accuracy: ", correct / cnt)
    return correct / cnt


def eval(model_name="gpt2", run_name='test', use_icl=False, pretrained_model=None, generate_images=False, k=1):
    subset_list = ['color', 'shape', 'material', 'component', 'others']

    vis_processors = AutoTokenizer.from_pretrained(
        model_name,
    )
    config = AutoConfig.from_pretrained(model_name)

    if 'gpt2' in model_name:
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            from_tf=bool(".ckpt" in model_name),
            config=config
        ).to('cuda')

    elif 'gemma' in model_name:
        num_added_tokens = vis_processors.add_tokens('<image>')
        config.image_token_index = vis_processors('<image>')[0].ids[1]
        config.ignore_index = -100
        config.pad_token_id = 0
        model = GemmaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, config=config).to(
            'cuda')

    elif 'llama' in model_name:
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, config=config).to(
            'cuda')

    elif 'opt' in model_name:
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, config=config).to(
            'cuda')

    if pretrained_model is not None:
        try:
            modules = torch.load(args.pretrained_model + '.bin')
            new_modules = modules.copy()
            for key in modules.keys():
                new_modules[key[7:]] = modules[key]
                del new_modules[key]

            model.load_state_dict(new_modules)

        except:
            model.load_state_dict(torch.load(args.pretrained_model + '.bin'))

    for subset in subset_list:
        print("Tested on the {} subset...".format(subset))
        results = []
        for idx in range(0, 5):  # prompt idx
            acc = _test(model, vis_processors, subset=subset, run_name=run_name, prompt_idx=idx, icl=use_icl, generate_images=generate_images, k=k)
            results.append(acc)
        with open("output/results/{}/{}/results.txt".format(run_name, subset), "w") as f:
            for idx, acc in enumerate(results):
                f.write("Accuracy for prompt{}: {} \n".format(idx, acc))
            avg = np.mean(results)
            std = np.std(results, ddof=1)
            f.write("Mean result: {}, Std result: {}".format(100 * avg, 100 * std))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser of ImageNetVC')
    parser.add_argument('--model_name', default='gpt2', type=str, help='pretrained model name or path')
    parser.add_argument('--run_name', default='test', type=str, help="store the results under this name")
    parser.add_argument('--pretrained_model', default=None, type=str, help="path to pretrained model")
    parser.add_argument('--use-icl', default=False, action='store_true', help='use in-context learning or not')
    parser.add_argument("--k", type=int, default=10, help="who many images to generate")
    parser.add_argument("--generate_images", type=bool, default=False, help="whether to generate images or no")
    args = parser.parse_args()
    eval(model_name=args.model_name, run_name=args.run_name, use_icl=args.use_icl,
         pretrained_model=args.pretrained_model, generate_images=args.generate_images, k=args.k)