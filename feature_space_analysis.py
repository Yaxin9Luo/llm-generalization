import torch
from transformers import GPT2Config
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
import random
from model_llm import MAE_GPT2_Classifier

class Args:
    def __init__(self):
        self.input_size = 224
        self.nb_classes = 10

def load_models():
    print("Loading models...")
    args = Args()
    pretrained_model = MAE_GPT2_Classifier(args, pretrained=True)
    random_model = MAE_GPT2_Classifier(args, pretrained=False)
    print("Models loaded successfully.")
    return pretrained_model, random_model

def build_cifar_transform(is_train, input_size=224):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=3),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    return transform

def load_cifar10(data_path='/root/autodl-tmp/data', input_size=224):
    print("Loading CIFAR-10 dataset...")
    transform = build_cifar_transform(is_train=True, input_size=input_size)
    dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    return dataset

def generate_data(dataset, num_samples=1000, random_label_ratio=1.0):
    print("Generating data...")
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    data = []
    true_labels = []
    random_labels = []
    
    num_classes = 10  # CIFAR-10 has 10 classes
    num_random_samples = int(num_samples * random_label_ratio)
    random_indices = random.sample(range(num_samples), num_random_samples)
    
    for i, idx in enumerate(tqdm(indices)):
        img, label = dataset[idx]
        data.append(img)
        true_labels.append(label)
        
        if i in random_indices:
            new_label = random.randint(0, num_classes - 1)
            while new_label == label:
                new_label = random.randint(0, num_classes - 1)
            random_labels.append(new_label)
        else:
            random_labels.append(label)
    
    print(f"\nRandomized {num_random_samples} labels out of {num_samples} in the dataset.")
    
    return torch.stack(data), torch.tensor(true_labels), torch.tensor(random_labels)

def extract_features(model, data):
    print("Extracting features...")
    features = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(torch.split(data, 32)):  # Process in batches
            gpt2_output = model.gpt2(inputs_embeds=model.patch_embed(batch)).last_hidden_state
            features.append(gpt2_output[:, -1, :].cpu().numpy())
    return np.vstack(features)

def visualize_features(features, true_labels, random_labels, method, model_type):
    print(f"Visualizing features using {method}...")
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    
    reduced_features = reducer.fit_transform(features)
    
    # Plot with true labels
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=true_labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f'{method.upper()} visualization of features ({model_type} model) - True Labels')
    plt.savefig(f'{method}_{model_type}_features_true_labels.png')
    plt.close()

    # Plot with random labels
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=random_labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f'{method.upper()} visualization of features ({model_type} model) - Random Labels')
    plt.savefig(f'{method}_{model_type}_features_random_labels.png')
    plt.close()
def visualize_raw_data(data, labels, method):
    print(f"Visualizing raw data using {method}...")
    flattened_data = data.view(data.size(0), -1).numpy()
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    
    reduced_data = reducer.fit_transform(flattened_data)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f'{method.upper()} visualization of raw CIFAR-10 data')
    plt.savefig(f'{method}_raw_data.png')
    plt.close()
def calculate_metrics(features, labels):
    print("Calculating metrics...")
    silhouette = silhouette_score(features, labels)
    intra_class_distance = []
    for label in np.unique(labels):
        class_features = features[labels == label]
        centroid = np.mean(class_features, axis=0)
        distances = np.linalg.norm(class_features - centroid, axis=1)
        intra_class_distance.append(np.mean(distances))
    avg_intra_class_distance = np.mean(intra_class_distance)
    return silhouette, avg_intra_class_distance

def main():
    pretrained_model, random_model = load_models()
    dataset = load_cifar10()
    data, true_labels, random_labels = generate_data(dataset)
    for method in ['tsne', 'pca']:
        visualize_raw_data(data, true_labels, method)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model.to(device)
    random_model.to(device)
    data = data.to(device)
    
    pretrained_features = extract_features(pretrained_model, data)
    random_features = extract_features(random_model, data)
    
    for method in ['tsne', 'pca']:
        visualize_features(pretrained_features, true_labels, random_labels, method, 'pretrained')
        visualize_features(random_features, true_labels, random_labels, method, 'random')
    
    pretrained_silhouette_true, pretrained_intra_class_true = calculate_metrics(pretrained_features, true_labels)
    pretrained_silhouette_random, pretrained_intra_class_random = calculate_metrics(pretrained_features, random_labels)
    random_silhouette_true, random_intra_class_true = calculate_metrics(random_features, true_labels)
    random_silhouette_random, random_intra_class_random = calculate_metrics(random_features, random_labels)
    
    print("\nMetrics:")
    print("Pretrained model:")
    print(f"  True Labels  - Silhouette Score: {pretrained_silhouette_true:.4f}, Avg Intra-class Distance: {pretrained_intra_class_true:.4f}")
    print(f"  Random Labels - Silhouette Score: {pretrained_silhouette_random:.4f}, Avg Intra-class Distance: {pretrained_intra_class_random:.4f}")
    print("Random model:")
    print(f"  True Labels  - Silhouette Score: {random_silhouette_true:.4f}, Avg Intra-class Distance: {random_intra_class_true:.4f}")
    print(f"  Random Labels - Silhouette Score: {random_silhouette_random:.4f}, Avg Intra-class Distance: {random_intra_class_random:.4f}")

if __name__ == "__main__":
    main()