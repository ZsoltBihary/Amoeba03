import torch
import time

# Define a simple model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(1000, 1000)

    def forward(self, x):
        # Simulate a long computation
        torch.cuda._sleep(10 * 10 ** 9)  # Simulate 2 seconds of GPU activity
        # time.sleep(2)  # Simulating a slow GPU operation
        return self.fc(x)

# Instantiate the model and move it to GPU
model = SimpleModel().cuda()

# Create some random data on CPU
data = torch.randn(1000, 1000)

# Pin the memory for faster transfers to and from GPU
pinned_data = data.pin_memory()

# Move data to GPU (asynchronous operation)
input_gpu = pinned_data.to('cuda', non_blocking=True)

# Simulate a long GPU computation
start = time.time()
output_gpu = model(input_gpu)

# In the meantime, CPU does some other work
print("CPU is free to do other operations...")
# Simulate CPU work
for i in range(5):
    print(f"CPU task {i+1} running...")
    time.sleep(0.5)  # Simulating CPU tasks

# Now, transfer the result from GPU to CPU asynchronously
output_cpu = output_gpu.to('cpu', non_blocking=True)  # Non-blocking CPU transfer

# Optionally synchronize to ensure all GPU tasks are done before using the result
# torch.cuda.synchronize()

end = time.time()

print("GPU computation and transfer finished.")
print(f"Total time taken: {end - start:.2f} seconds")
