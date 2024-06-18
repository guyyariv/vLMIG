import argparse
import logging
import math
import os
import random
from diffusers import AutoPipelineForText2Image
import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    SchedulerType,
    get_scheduler,
)
from transformers.utils.versions import require_version
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import crop
from transformers import AutoTokenizer
from load_datasets import load_wiki, load_laion_220, load_cropped_vg_regions
from nltk.tokenize import sent_tokenize

from models.gpt2.modeling_gpt2_fusion_layer import GPT2LMHeadModel
from models.gemma.modeling_gemma_fusion_layer import GemmaForCausalLM
from models.llama3.modeling_llama3_fusion_layer import LlamaForCausalLM
from models.bert.modeling_bert_fusion_layer import BertForMaskedLM
from models.opt.modeling_opt_fusion_layer import OPTForCausalLM

# check_min_version("4.36.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# Transformation to convert PIL Images to tensors
to_tensor = transforms.ToTensor()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="complete here")
    parser.add_argument("--run_name", type=str, default='test')
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="The name of the dataset to use.")
    parser.add_argument("--max_train_samples", type=str, default=None,
                        help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.")
    parser.add_argument("--model_name_or_path", type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models.", required=False)
    parser.add_argument("--config_name", type=str, default=None,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                 "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--model_type", type=str, default=None, help="Model type to use if training from scratch.",
                        choices=MODEL_TYPES)
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.")
    parser.add_argument("--trust_remote_code", type=bool, default=False,
                        help="Whether or not to allow for custom models defined on the Hub in their own modeling files. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.")
    parser.add_argument("--checkpointing_steps", type=str, default=None,
                        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="If the training should continue from a checkpoint folder.")
    parser.add_argument("--with_tracking", action="store_true",
                        help="Whether to enable experiment trackers for logging.")
    parser.add_argument("--report_to", type=str, default="wandb",
                        help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`, `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. Only applicable when `--with_tracking` is passed.')
    parser.add_argument("--low_cpu_mem_usage", action="store_true",
                        help="It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. If passed, LLM loading time and RAM consumption will be benefited.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Path to pretrained weights")
    parser.add_argument("--max_elements", type=int, default=None,
                        help="Choose how many elements will be taken from each of the datasets, if None, the full training will be trained")
    parser.add_argument("--shuffle_data", action="store_true",
                        help="Whether to shuffle the training set")
    parser.add_argument("--run_bf16", action="store_true",
                        help="Whether to train with bfloat16")

    args = parser.parse_args()

    return args


def collate_fn_gemma(examples):
    image_col = 'image_path'
    images = [example[image_col] for example in examples]
    input_ids = torch.tensor([example["input_ids"]for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)

    non_zero_start = (input_ids != 0).long().argmax(dim=1)
    min_index = non_zero_start.min() - 1
    input_ids = input_ids[:, min_index:]
    attention_mask = attention_mask[:, min_index:]

    input_ids[input_ids[:, 1] != 0, 1] = 2
    input_ids[:, 0] = 0
    attention_mask[:, 0] = 0

    if 'crop' in examples[0]:
        crop = [example["crop"] for example in examples]
        return {
            "image_path": images,
            "input_ids": input_ids,
            "labels": input_ids,
            "attention_mask": attention_mask,
            "crop": crop,
        }

    return {
        "image_path": images,
        "input_ids": input_ids,
        "labels": input_ids,
        "attention_mask": attention_mask,
    }


def collate_fn_bert(examples):

    def mask_tokens(inputs, mlm_probability=0.20):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = torch.zeros_like(inputs).bool()
        special_tokens_mask[torch.where((inputs == 101) | (inputs == 102) | (inputs == 0))] = True

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = 103

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(30522, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    # pixel_values = torch.stack([example["pixel_values"] for example in examples])
    # crop = [example["crop"] for example in examples]
    image_path = [example["image_path"] for example in examples]
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)

    max_index = 100
    input_ids = input_ids[:, :max_index]
    attention_mask = attention_mask[:, :max_index]
    input_ids, labels = mask_tokens(input_ids)

    # Mask of elements not allowed to be replaced
    mask = ((input_ids != 101) & (input_ids != 102) & (input_ids != 0))

    # Check if 103 exists in any vector
    exists_103 = torch.any(input_ids == 103, dim=1, keepdim=True)

    # Broadcast exists_103 to match the shape of mask
    exists_103 = exists_103.expand_as(mask)

    # Create a mask of where to replace
    replace_mask = (~exists_103.unsqueeze(1) & mask)

    # Choose random index for each row where replacement is needed
    row_indices = torch.nonzero(replace_mask, as_tuple=False)[:, 0]
    unique_row_indices = row_indices.unique()
    replace_indices = torch.stack(
        [unique_row_indices, torch.randint(2, 10, size=(len(unique_row_indices),))], dim=1)

    # Replace with 103
    labels[replace_indices[:, 0], replace_indices[:, 1]] = input_ids[replace_indices[:, 0], replace_indices[:, 1]]
    input_ids[replace_indices[:, 0], replace_indices[:, 1]] = 103

    return {
        "image_path": image_path,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        # "crop": crop,
    }


def collate_fn_gpt2(examples):
    image_col = 'image_path'
    images = [example[image_col] for example in examples]
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)

    non_zero_start = (input_ids == 50256).long().argmax(dim=1)
    max_index = non_zero_start.max()
    input_ids = input_ids[:, :max_index]
    attention_mask = attention_mask[:, :max_index]

    if 'crop' in examples[0]:
        crop = [example["crop"] for example in examples]
        return {
            "image_path": images,
            "input_ids": input_ids,
            "labels": input_ids,
            "attention_mask": attention_mask,
            "crop": crop,
        }

    return {
        "image_path": images,
        "input_ids": input_ids,
        "labels": input_ids,
        "attention_mask": attention_mask,
    }


def collate_fn_opt(examples):
    image_col = 'image_path'
    images = [example[image_col] for example in examples]
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)

    non_zero_start = (input_ids == 1).long().argmax(dim=1)
    max_index = non_zero_start.max()
    input_ids = input_ids[:, :max_index]
    attention_mask = attention_mask[:, :max_index]

    if 'crop' in examples[0]:
        crop = [example["crop"] for example in examples]
        return {
            "image_path": images,
            "input_ids": input_ids,
            "labels": input_ids,
            "attention_mask": attention_mask,
            "crop": crop,
        }

    return {
        "image_path": images,
        "input_ids": input_ids,
        "labels": input_ids,
        "attention_mask": attention_mask,
    }


def collate_fn_llama(examples):
    image_col = 'image_path'
    images = [example[image_col] for example in examples]
    input_ids = torch.tensor([example["input_ids"][:100] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"][:100] for example in examples], dtype=torch.long)

    non_zero_start = (input_ids == 128001).long().argmax(dim=1)
    max_index = non_zero_start.max()
    input_ids = input_ids[:, :max_index]
    attention_mask = attention_mask[:, :max_index]

    if 'crop' in examples[0]:
        crop = [example["crop"] for example in examples]
        return {
            "image_path": images,
            "input_ids": input_ids,
            "labels": input_ids,
            "attention_mask": attention_mask,
            "crop": crop,
        }

    return {
        "image_path": images,
        "input_ids": input_ids,
        "labels": input_ids,
        "attention_mask": attention_mask,
    }


def main():
    args = parse_args()

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    model_dtype = torch.bfloat16 if args.run_bf16 else torch.float

    accelerator.wait_for_everyone()

    if "gpt2" in args.model_name_or_path:
        model = GPT2LMHeadModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=model_dtype,
        )
        tokenizer.pad_token = tokenizer.eos_token

    if 'bert' in args.model_name_or_path:
        model = BertForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=model_dtype,
        )

    elif 'gemma' in args.model_name_or_path:
        num_added_tokens = tokenizer.add_tokens('<image>')
        config.image_token_index = tokenizer('<image>')[0].ids[1]
        config.ignore_index = -100
        config.pad_token_id = 0
        model = GemmaForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype, config=config)

    elif 'llama' in args.model_name_or_path:
        config.ignore_index = -100
        config.pad_token_id = 128001
        tokenizer.pad_token = tokenizer.eos_token
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype, config=config)

    elif 'opt' in args.model_name_or_path:
        config.ignore_index = -100
        model = OPTForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype, config=config)

    processor = model.processor

    if args.pretrained_model is not None:
        modules = torch.load(f'{args.pretrained_model}.bin')
        new_modules = modules.copy()
        for key in modules.keys():
            new_modules[key[7:]] = modules[key]
            del new_modules[key]

        model.load_state_dict(new_modules)
        del new_modules, modules

    dataset_lst = []

    if 'wiki' in args.dataset_name:
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16)
        pipe.to("cuda")

    if 'wiki' in args.dataset_name:
        dataset_lst.append(load_wiki(args, tokenizer))

    if 'laion_220' in args.dataset_name:
        short = 'short' in args.dataset_name
        dataset_lst.append(load_laion_220(args, tokenizer, short))

    if 'cropped_vg' in args.dataset_name:
        dataset_lst.append(load_cropped_vg_regions(args, tokenizer))

    if args.max_elements is not None:
        if args.shuffle_data:
            dataset_lst = [
                dataset.shuffle(seed=42).select(range(min(len(dataset), args.max_elements)))
                for dataset in dataset_lst
            ]

        else:
            dataset_lst = [dataset_lst[i].select(range(min(len(dataset_lst[i]),
                                                           args.max_elements))) for i in range(len(dataset_lst))]

    train_dataset = datasets.concatenate_datasets(
        dataset_lst
    )

    del dataset_lst

    # eval_dataset = load_coco(args, tokenizer, valid_set=True)

    if "gemma" in args.model_name_or_path:
        fn = collate_fn_gemma

    elif "llama" in args.model_name_or_path:
        fn = collate_fn_llama

    elif "bert" in args.model_name_or_path:
        fn = collate_fn_bert

    elif "gpt2" in args.model_name_or_path:
        fn = collate_fn_gpt2

    elif "opt" in args.model_name_or_path:
        fn = collate_fn_opt

    dataloader_params = {
        "batch_size": args.per_device_train_batch_size,
        "collate_fn": fn,
        "num_workers": 0,
        "shuffle": args.shuffle_data
    }
    train_dataloader = DataLoader(train_dataset, **dataloader_params)

    model.requires_grad_(False)

    optimizer_grouped_parameters = []

    model.mm_proj.requires_grad_(True)
    optimizer_grouped_parameters += list(model.mm_proj.parameters())

    model.fusion_layer.requires_grad_(True)
    optimizer_grouped_parameters += list(model.fusion_layer.parameters())

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    # model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    # )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("vglm", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    total_loss = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):

                images = []
                indices = []
                for i in range(len(batch['image_path'])):
                    if batch['image_path'][i] is None:
                        indices.append(i)
                        images.append(None)  # This will be replaced by the generated image
                    else:
                        image = read_image(batch['image_path'][i], mode=ImageReadMode.RGB)
                        if "crop" in batch and batch['crop'][i] is not None:
                            x, y, width, height = [max(0, val) for val in batch['crop'][i]]
                            image = crop(image, top=y, left=x, height=height, width=width)
                        images.append(image)


                # Decode prompts for missing images
                prompts = tokenizer.batch_decode(batch['input_ids'][indices], skip_special_tokens=True)
                prompts = [sent_tokenize(prompt) for prompt in prompts]
                for i in range(len(prompts)):
                    if not prompts[i]:
                        prompts[i].append('random')
                prompts = [random.choice(sentences) for sentences in prompts]
                generated_images = []

                # Generate images where originals were missing
                # Adjust the function call according to your specific 'pipe' function
                if prompts:
                    generated_images = \
                    pipe(prompts, num_inference_steps=1, guidance_scale=0.0)[0]


                # Insert generated images into the correct positions in the images list
                for index, gen_image in zip(indices, generated_images):
                    tensor_image = to_tensor(gen_image)
                    tensor_image = (tensor_image * 255).type(torch.uint8)
                    images[index] = tensor_image

                del batch['image_path']
                if "crop" in batch:
                    del batch['crop']

                batch['pixel_values'] = processor(images=images, return_tensors="pt").pixel_values.to(
                    accelerator.unwrap_model(model).dtype).to(accelerator.unwrap_model(model).device)

                # Model forward pass
                outputs = model(**batch)
                loss = outputs.loss

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()

                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     params_to_clip = (
                #         itertools.chain(accelerator.unwrap_model(model).fusion_layer.parameters(),
                #                         accelerator.unwrap_model(model).img_proj.parameters())
                #     )
                #     accelerator.clip_grad_norm_(params_to_clip, 1)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if args.with_tracking:
                    accelerator.log(
                        {
                            "train_loss": loss.item(),
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )
                    total_loss = 0

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            torch.save(model.state_dict(), f'{args.output_dir}/{args.run_name}_{epoch}.bin')


if __name__ == "__main__":
    main()
