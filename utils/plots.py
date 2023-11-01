import matplotlib.pyplot as plt

# Sample data: Replace this with your actual accuracy results
dataset_names = ["Raabin", "Matek", "Acevedo"]
method_names = ["Naive", "EWC", "Replay"]
iteration_results = [
    # For Method Naive
    [
        [0.90, 0.77, 0.51],  # Dataset 1
        [0.43, 0.82, 0.91],  # Dataset 2
        [0.36, 0.39, 0.99],  # Dataset 3
    ],
    # For Method B
    [
        [0.92, 0.73, 0.60],  # Dataset 1
        [0.57, 0.96, 0.83],  # Dataset 2
        [0.40, 0.40, 0.99],  # Dataset 3
    ],
    # For Method C
    [
        [0.89, 0.92, 0.89],  # Dataset 1
        [0.76, 0.95, 0.95],  # Dataset 2
        [0.35, 0.37, 0.99],  # Dataset 3
    ]
]

# Create a plot for all datasets and methods together
for i, dataset in enumerate(dataset_names):
    plt.figure()
    for j, method in enumerate(method_names):
        plt.plot(range(1, 4), iteration_results[j][i], marker='o', label=method)
    plt.title(f'Accuracy for {dataset}')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, 4))
    plt.grid()
    plt.legend()
    plt.savefig(f'{dataset.replace(" ", "_")}_accuracy.png', format='png')
    plt.close()