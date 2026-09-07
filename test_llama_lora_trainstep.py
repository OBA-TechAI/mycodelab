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
MAX_LENGTH = 1024

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
model.config.use_cache = False

model.gradient_checkpointing_enable()

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

model.enable_input_require_grads()
model.train()

text = (
    "Patient prior diagnosis history. "
    "Visit 1 diagnoses: Type 2 diabetes mellitus; "
    "hypertension; chronic kidney disease. "
) * 100

encoded = tokenizer(
    text,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

input_ids = encoded["input_ids"].cuda()
attention_mask = encoded["attention_mask"].cuda()

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4
)

optimizer.zero_grad()

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask
)

logits = outputs.logits.squeeze(-1)

label = torch.tensor(
    [1.0],
    device="cuda"
)

loss = torch.nn.functional.binary_cross_entropy_with_logits(
    logits,
    label
)

loss.backward()
optimizer.step()

print("Forward/backward step: SUCCESS")
print("Loss:", float(loss.detach().cpu()))

print(
    "Peak GPU memory:",
    round(
        torch.cuda.max_memory_allocated() / 1024**3,
        2
    ),
    "GB"
)
