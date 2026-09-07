import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./Llama-3.2-3B-Instruct"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Loading Llama model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True
).to("cuda")

model.eval()

print("Model loaded successfully.", flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)

messages = [
    {
        "role": "user",
        "content": "Explain diabetic nephropathy in one simple sentence."
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

print("Running inference...", flush=True)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(
    outputs[0][inputs.shape[-1]:],
    skip_special_tokens=True
)

print("\nLlama response:")
print(response)

print("\nGPU memory allocated:",
      round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")

print("Llama inference test: SUCCESS")
