import datasets
import os
from datasets import load_dataset
from itertools import chain


def load_laion_220(args, tokenizer, short=False):

    path_to_images = "/cs/labs/Academic/dataset/COCO"
    caption_column = 'caption'
    image_column = 'image_path'
    dataset_path = f"datasets/laion_220_{args.model_name_or_path}.hf"
    if short:
        dataset_path = f"datasets/laion_220_short_{args.model_name_or_path}.hf"
        caption_column = 'short_caption'

    def tokenize_laion_220(examples):
        captions = list(examples[caption_column])
        image_path = [os.path.join(path_to_images, file_name[30:]) for file_name in examples["url"]]

        text_inputs = tokenizer(captions, max_length=args.max_seq_length, padding="max_length", truncation=True)

        examples["input_ids"] = text_inputs['input_ids']
        examples["attention_mask"] = text_inputs["attention_mask"]
        examples[image_column] = image_path
        return examples

    try:
        laion_220_dataset = datasets.load_from_disk(dataset_path)

    except:
        dataset = load_dataset("laion/220k-GPT4Vision-captions-from-LIVIS")
        column_names = dataset["train"].column_names

        laion_220_dataset = dataset["train"]

        if args.max_train_samples is not None:
            max_train_samples = min(len(laion_220_dataset), args.max_train_samples)
            laion_220_dataset = laion_220_dataset.select(range(max_train_samples))

        laion_220_dataset = laion_220_dataset.map(
            function=tokenize_laion_220,
            batched=True,
            remove_columns=[col for col in column_names],
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        laion_220_dataset.save_to_disk(dataset_path)

    return laion_220_dataset


def load_wiki(args, tokenizer):
    dataset_path = f"datasets/wiki_{args.wiki_num}_connected_{args.model_name_or_path}.hf"

    try:
        lm_datasets = datasets.load_from_disk(dataset_path)

    except:
        raw_dataset = load_dataset("wikitext", f'wikitext-{args.wiki_num}-raw-v1')['train']
        column_names = raw_dataset.column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        # block_size = tokenizer.model_max_length
        block_size = 447

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        tokenized_datasets = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            result['image_path'] = [None] * len(result['input_ids'])
            return result

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

        lm_datasets.save_to_disk(dataset_path)

    return lm_datasets


def load_cropped_vg_regions(args, tokenizer):
    dataset_path = f"datasets/cropped_vg_regions_{args.model_name_or_path}.hf"
    try:
        vg_train_dataset = datasets.load_from_disk(dataset_path)

    except:
        vg_dataset = load_dataset("visual_genome", "region_descriptions_v1.2.0")
        caption_column = 'regions'

        def tokenize_captions_visual_genome(examples):
            captions = []
            image_path = []
            crop = []
            for idx, e in enumerate(examples[caption_column]):

                # path = examples['image'][idx].filename
                image = examples['image'][idx]
                try:
                    path = examples['image'][idx].filename
                except:
                    path = f"datasets/visual_genome_regions/{e[0]['image_id']}.png"
                    image.info.pop('icc_profile', None)
                    image.save(path)

                for scene in e:
                    captions.append(scene['phrase'])
                    crop.append((scene['x'], scene['y'], scene['width'], scene['height']))
                    image_path.append(path)

            text_inputs = tokenizer(captions, max_length=args.max_seq_length, padding="max_length", truncation=True)

            examples["input_ids"] = text_inputs['input_ids']
            examples["attention_mask"] = text_inputs['attention_mask']
            examples["image_path"] = image_path
            examples["crop"] = crop
            return examples

        vg_train_dataset = vg_dataset["train"]

        if args.max_train_samples is not None:
            max_train_samples = min(len(vg_train_dataset), args.max_train_samples)
            vg_train_dataset = vg_train_dataset.select(range(max_train_samples))

        column_names_vg = vg_train_dataset.column_names

        vg_train_dataset = vg_train_dataset.map(
            function=tokenize_captions_visual_genome,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=[col for col in column_names_vg],
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        vg_train_dataset.save_to_disk(dataset_path)

    return vg_train_dataset
