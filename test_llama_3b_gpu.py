import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./Llama-3.2-3B-Instruct"

print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("GPU not available")

print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map={"": 0},
    local_files_only=True,
    low_cpu_mem_usage=True
)

print("Model loaded: SUCCESS")
print("Model type:", model.config.model_type)
print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
print(
    "GPU memory allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)
