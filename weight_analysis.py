import torch
from transformers import GPT2Model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from tqdm import tqdm

def load_models():
    # 加载预训练模型
    pretrained_model = GPT2Model.from_pretrained("gpt2-medium")
    
    # 加载随机初始化模型
    config = pretrained_model.config
    random_model = GPT2Model(config)
    
    return pretrained_model, random_model

def extract_weights(model):
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data.cpu().numpy().flatten())
    return np.concatenate(weights)

def plot_weight_distribution(pretrained_weights, random_weights, layer_name):
    plt.figure(figsize=(12, 6))
    
    sns.kdeplot(pretrained_weights, label='Pretrained', shade=True)
    sns.kdeplot(random_weights, label='Random Init', shade=True)
    
    plt.title(f'Weight Distribution - {layer_name}')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'weight_dist_{layer_name}.png')
    plt.close()

def compare_layer_statistics(pretrained_model, random_model):
    stats_data = []
    
    for (name1, param1), (name2, param2) in tqdm(zip(pretrained_model.named_parameters(), random_model.named_parameters()), desc="Comparing layers"):
        if 'weight' in name1:
            pre_weights = param1.data.cpu().numpy().flatten()
            rand_weights = param2.data.cpu().numpy().flatten()
            
            stats_data.append({
                'layer': name1,
                'pretrained_mean': np.mean(pre_weights),
                'random_mean': np.mean(rand_weights),
                'pretrained_std': np.std(pre_weights),
                'random_std': np.std(rand_weights),
                'pretrained_range': np.ptp(pre_weights),
                'random_range': np.ptp(rand_weights),
                'ks_statistic': stats.ks_2samp(pre_weights, rand_weights).statistic
            })
    
    return stats_data

def plot_layer_statistics(stats_data):
    layers = [stat['layer'] for stat in stats_data]
    ks_stats = [stat['ks_statistic'] for stat in stats_data]
    
    plt.figure(figsize=(15, 8))
    plt.bar(layers, ks_stats)
    plt.title('KS Statistic for Each Layer')
    plt.xlabel('Layer')
    plt.ylabel('KS Statistic')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('ks_statistic_by_layer.png')
    plt.close()

def main():
    pretrained_model, random_model = load_models()
    print("model loaded successfully!!")
    pretrained_weights = extract_weights(pretrained_model)
    random_weights = extract_weights(random_model)
    
    plot_weight_distribution(pretrained_weights, random_weights, 'All Layers')
    
    stats_data = compare_layer_statistics(pretrained_model, random_model)
    plot_layer_statistics(stats_data)
    
    # 打印每层的统计信息
    for stat in stats_data:
        print(f"Layer: {stat['layer']}")
        print(f"  Pretrained - Mean: {stat['pretrained_mean']:.4f}, Std: {stat['pretrained_std']:.4f}, Range: {stat['pretrained_range']:.4f}")
        print(f"  Random    - Mean: {stat['random_mean']:.4f}, Std: {stat['random_std']:.4f}, Range: {stat['random_range']:.4f}")
        print(f"  KS Statistic: {stat['ks_statistic']:.4f}")
        print()

if __name__ == "__main__":
    main()