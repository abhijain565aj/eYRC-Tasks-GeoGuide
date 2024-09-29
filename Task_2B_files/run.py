from tqdm import tqdm
import time

total_iterations = 100

# Create a customized loading bar
for i in tqdm(range(total_iterations), desc="Processing", ncols=100, bar_format="{l_bar}{bar}{r_bar}"):
    time.sleep(10)

print("Task completed!")
