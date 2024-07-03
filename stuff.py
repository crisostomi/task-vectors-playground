import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
else:
    print("CUDA is not available.")

# Example of using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example tensor on GPU
tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
print(tensor)
