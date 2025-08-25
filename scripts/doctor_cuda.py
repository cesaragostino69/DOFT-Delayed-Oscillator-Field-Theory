import torch, os, sys
print("python:", sys.executable)
print("torch:", getattr(torch, "__version__", "NA"))
print("cuda available:", torch.cuda.is_available())
print("torch cuda version (built):", torch.version.cuda if hasattr(torch.version,'cuda') else None)
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f" [{i}] name:", torch.cuda.get_device_name(i))
print("USE_GPU env:", os.getenv("USE_GPU"))