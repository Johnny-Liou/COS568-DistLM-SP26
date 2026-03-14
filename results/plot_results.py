import re
import matplotlib.pyplot as plt
import os

def parse_log(file_path):
    steps = []
    losses = []
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist.")
        return steps, losses
        
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r'Step (\d+), Loss: ([0-9.]+)', line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return steps, losses

def plot_figures():
    # Figure 1: Rank 0 Loss Curve comparison
    plt.figure(figsize=(10, 6))
    
    tasks = {
        'Gather-Scatter': 'task2a',
        'All Reduce': 'task2b',
        'DDP': 'task3'
    }
    
    for label, task in tasks.items():
        path = os.path.join('results', task, 'node0.txt')
        steps, losses = parse_log(path)
        if steps:
            plt.plot(steps, losses, label=label, linewidth=2, alpha=0.7)
            
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss on Rank 0 (Comparison)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/loss_comparison_rank0.png')
    print("Generated results/loss_comparison_rank0.png")
    
    # Figure 2: Different Ranks Loss Curve (using Task 2a as example)
    plt.figure(figsize=(10, 6))
    task = 'task2a'
    for rank in range(4):
        path = os.path.join('results', task, f'node{rank}.txt')
        steps, losses = parse_log(path)
        if steps:
            plt.plot(steps, losses, label=f'Rank {rank}', linewidth=2, alpha=0.7, linestyle='--' if rank > 1 else '-')

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Across Ranks (Gather-Scatter)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/loss_comparison_ranks.png')
    print("Generated results/loss_comparison_ranks.png")

if __name__ == "__main__":
    plot_figures()
