import matplotlib.pyplot as plt

# Data
models = ['Llama2', 'Llama2-LoRA', 'Llama2-Freeze', 'ESGLlama', 'FinLlama']
methods = ['Zero Shot', 'Zero Shot w/ CoT', 'One Shot', 'One Shot w/ CoT', 'ICL', 'ICL w/ CoT']
precision_values = [
    [0.5778, 0.5527, 0.6012, 0.5370, 0.6687, 0.6794],  # Llama2
    [0.6928, 0.6381, 0.5265, 0.5646, 0.6148, 0.6213],  # Llama2-LoRA
    [0.5741, 0.5480, 0.6085, 0.6168, 0.6611, 0.6749],  # Llama2-Freeze
    [0.5770, 0.5502, 0.6106, 0.6064, 0.6738, 0.6746],  # ESGLlama
    [0.5766, 0.5665, 0.6139, 0.5724, 0.6698, 0.6797]   # FinLlama
]

# nine-class
# precision_values = [
#     [0.5875, 0.5826, 0.5049, 0.4314, 0.6108, 0.6164],  # Llama2
#     [0.5681, 0.5180, 0.6256, 0.5751, 0.6242, 0.6544],  # Llama2-LoRA
#     [0.5911, 0.5799, 0.5258, 0.4922, 0.6285, 0.5719],  # Llama2-Freeze
#     [0.5866, 0.5914, 0.5138, 0.4785, 0.6201, 0.5773],  # ESGLlama
#     [0.5608, 0.5750, 0.5219, 0.4886, 0.6168, 0.6654]   # FinLlama
# ]


# Colors and markers
colors = ['#2ca02c', '#ff7f0e', '#1f77b4', 'darkviolet', 'firebrick']
markers = ['o', 'o', 'o', 's', '^']

# Plotting
plt.figure(figsize=(10, 6))
for i, (model, precision) in enumerate(zip(models, precision_values)):
    plt.plot(methods, precision, marker=markers[i], color=colors[i], label=model)

plt.title('Precision Score Variation across Different Methods')
plt.xlabel('Method')
plt.ylabel('Precision Score')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and show plot
plt.savefig('four_class_precision_variation_plot.png')
plt.show()
