import sys
import torch
import transformers
import peft
import accelerate
import numpy
import pandas
import sklearn
import pyarrow

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)
print("Accelerate:", accelerate.__version__)
print("NumPy:", numpy.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("A30 GPU is not available")

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA build:", torch.version.cuda)
print("BF16 supported:", torch.cuda.is_bf16_supported())

# Test the BF16 operation that failed in the old environment.
x = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
y = torch.triu(x, diagonal=1)
torch.cuda.synchronize()

print("BF16 triangular-mask test: SUCCESS")
print("Llama environment GPU check: SUCCESS")
