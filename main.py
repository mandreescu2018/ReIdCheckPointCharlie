import torch

# Create a PyTorch tensor with integers
# integer_tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)  # You can use torch.int64 or other integer types as needed
integer_tensor = torch.randint(0, 200, size=(1, 128), dtype=torch.int32)
# Print the integer tensor
print(integer_tensor)

print("main entrypoint")
