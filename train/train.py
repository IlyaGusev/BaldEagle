import json
import os
import torch
import wandb
import random

from safetensors import safe_open

from transformers.models.llama.configuration_llama import LlamaConfig

from transformers import AutoTokenizer, TrainingArguments

from modules.model.llama_eagle import LlamaForCausalLMEagle
from modules.data.data import (
    EagleLocalDataset,
    DataCollatorWithPadding,
    AddUniformNoise,
    list_local_files,
)
from modules.trainer.trainer import EagleTrainer

wandb.init(project="BaldEagle")
wandb_run_name = wandb.run.name

path = "../saiga/models/SAINEMO-reMIX"

# -------------------------------- Load original Llama weights --------------------------------

with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
    index_json = json.loads(f.read())
    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
    lm_head_path = index_json["weight_map"]["lm_head.weight"]

with safe_open(os.path.join(path, emb_path), framework="pt", device="cpu") as f:
    tensor_slice = f.get_slice("model.embed_tokens.weight")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim]

with safe_open(os.path.join(path, lm_head_path), framework="pt", device="cpu") as f:
    lm_head_weights = f.get_slice("lm_head.weight")[:, :]


# -------------------------------- Create draft model + tokenizer + head --------------------------------

tokenizer = AutoTokenizer.from_pretrained(path)
tokenizer.pad_token = tokenizer.eos_token

model_args = LlamaConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_dim,
    intermediate_size=14336,
    head_dim=128,
    num_hidden_layers=1,
    bos_token_id=1,
    eos_token_id=2,
    num_key_value_heads=8,
    num_attention_heads=32,
    tie_word_embeddings=False,
    max_position_embeddings=16384,
)

draft_model = LlamaForCausalLMEagle(model_args)
draft_model.load_embedding_weights(tensor)
draft_model.to("cuda:0")
draft_model = draft_model.to(torch.bfloat16)
draft_model.embed_tokens.weight.requires_grad = False

# Load head
head = torch.nn.Linear(model_args.hidden_size, model_args.vocab_size, bias=False)
with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
    index_json = json.loads(f.read())
    head_path = index_json["weight_map"]["lm_head.weight"]
with safe_open(os.path.join(path, head_path), framework="pt", device="cpu") as f:
    tensor_slice = f.get_slice("lm_head.weight")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim].float()

head.weight.data = tensor
head.to("cuda:0")
head.eval()

# -------------------------------- Load data --------------------------------

train_data_paths = list_local_files("../saiga/vllm_train_eagle")
eval_data_paths = list_local_files("../saiga/vllm_val_eagle")

eagle_train_dataset = EagleLocalDataset(
    train_data_paths, transform=AddUniformNoise(std=0.5)
)
eagle_test_dataset = EagleLocalDataset(eval_data_paths)

eagle_collator = DataCollatorWithPadding()

# -------------------------------- Train --------------------------------

training_args = TrainingArguments(
    output_dir=f"../saiga/models/{wandb_run_name}/",
    num_train_epochs=20,
    gradient_accumulation_steps=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    remove_unused_columns=False,
    bf16=True,
    fp16=False,
    dataloader_num_workers=4,
    warmup_ratio=0.01,
    learning_rate=1e-4,  # 1e-3
    lr_scheduler_type="constant",  # Placeholder, we override it in the trainer
    max_grad_norm=0.5,  # 1
    adam_beta1=0.9,  # 0.9
    adam_beta2=0.95,  # 0.999
    weight_decay=0.01,
    eval_strategy="steps",
    logging_steps=32,
    eval_steps=512,
    save_strategy="steps",
    save_steps=512,
    save_total_limit=1,
)

trainer = EagleTrainer(
    model=draft_model,
    head=head,
    args=training_args,
    train_dataset=eagle_train_dataset,
    eval_dataset=eagle_test_dataset,
    data_collator=eagle_collator,
    min_lr_ratio=0.5,  # Custmer lr scheduler param
)

trainer.train()
