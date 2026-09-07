import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)

MODEL_PATH = "./Llama-3.2-3B-Instruct"

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=1,
    torch_dtype=torch.float16,
    device_map={"": 0},
    local_files_only=True,
    low_cpu_mem_usage=True
)

model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    bias="none"
)

model = get_peft_model(
    model,
    lora_config
)

trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total = sum(
    p.numel()
    for p in model.parameters()
)

print("LoRA setup: SUCCESS")
print("Total parameters:", f"{total:,}")
print("Trainable parameters:", f"{trainable:,}")
print(
    "Trainable percentage:",
    round(100 * trainable / total, 4),
    "%"
)

print(
    "GPU memory allocated:",
    round(
        torch.cuda.memory_allocated() / 1024**3,
        2
    ),
    "GB"
)
