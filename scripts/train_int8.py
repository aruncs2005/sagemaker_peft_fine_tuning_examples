import gc
import os
import sys
import threading

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
import functools
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)
from torch import nn
import time

import copy
from itertools import chain

import psutil
from datasets import load_dataset
from tqdm import tqdm
from utils import parse_args
import json
from peft import LoraConfig, TaskType, get_peft_model

from peft import prepare_model_for_int8_training



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
)


def main():
    args = parse_args()


    text_column = "question"
    label_column = "answer"


    accelerator = Accelerator()

    set_seed(args.seed)

    print("*********************************************")
    print(f"Total number of processes in the training {accelerator.num_processes}")
    print(f"current device {accelerator.device}")
    print(f"Accelerate local rank {accelerator.local_process_index} Accelerate rank {accelerator.process_index} Accelerate world size {accelerator.num_processes}")
    print(f"Torchrun local rank {args.local_rank} torchrun rank {args.rank} torchrun world size {args.world_size}")
    print("*********************************************")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            print(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            print(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    dataset = load_dataset(
        'csv', data_files={
        "train": args.train_file,
        "validation": args.validation_file,
        })
    

    def preprocess_function(examples):

        instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: "
        response_prefix = "### Response: "
        inputs = [instruction + prompt + response_prefix + response+tokenizer.eos_token for prompt,response in zip(examples[text_column],examples[label_column])]

        model_inputs = tokenizer(inputs)
        model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])
        return model_inputs
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    with accelerator.main_process_first():
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        processed_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                desc=f"Grouping texts in chunks of {block_size}",
            )
     
    accelerator.wait_for_everyone()

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    train_sampler = torch.utils.data.DistributedSampler(
                train_dataset,
                shuffle=True,
                seed=args.seed,
                rank=args.rank,
                num_replicas=args.world_size,
                drop_last=True,
            )
    
    eval_sampler = torch.utils.data.DistributedSampler(
                eval_dataset,
                shuffle=True,
                seed=args.seed,
                 rank=args.rank,
                num_replicas=args.world_size,
                drop_last=True,
            )


    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,sampler=eval_sampler, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size, pin_memory=True
    )

    print(next(iter(train_dataloader)))

    # creating model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,cache_dir=f"/tmp",load_in_8bit=True)
    model = prepare_model_for_int8_training(model)

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)


    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)


    #lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_train_epochs),
    )

    
    model,train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
    )
    accelerator.print(model)

    start = time.time()
    total_steps = 0
    for epoch in range(args.num_train_epochs):

        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            #batch = {k: v.to(device) for k, v in batch.items()}
            step_start = time.time()
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_steps += 1
            time_elapsed = time.time() - start
            step_time = time.time() - step_start
            if step > 50: # stop for testing
                break
            if accelerator.is_main_process:
                #print(torch.cuda.memory_summary(device=0))
                print(
                    f"({int(time_elapsed)}s), Batch {total_steps - 1} Loss: {loss.detach().float()}"
                )

           

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        model.eval()
        eval_preds = []
        eval_loss = 0
        for estep, batch in enumerate(tqdm(eval_dataloader)):
            #batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
                 )
            if estep > 50: # for testing
                break
        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)

        accelerator.print(f"{epoch=}: {eval_ppl=} {eval_epoch_loss=}")

    accelerator.wait_for_everyone()

    #save the adapter configs
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(os.path.join(args.model_dir, "peft_model/"))

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
