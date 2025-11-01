from registries import register
import torch
import numpy as np

def convert_samples_to_torch(samples):
    """transfer numpy samples to torch samples"""
    torch_samples = []
    for sample in samples:
        torch_sample = {}
        for key, value in sample.items():
            if key == "A":
                edge_index, edge_weight = value
                torch_sample[key] = value# a 保留因为要在读取的时候改为稠密矩阵
            if key == "y_cls":
                    torch_sample[key] = torch.tensor(value).long()
            elif key == "y_reg":
                    torch_sample[key] = torch.tensor(value).float()
            elif isinstance(value, np.ndarray): # y_cls transfer to long
                torch_sample[key] = torch.from_numpy(value).float()
            else:
                torch_sample[key] = value
        torch_samples.append(torch_sample)
    return torch_samples

def load_samples_from_npz(npz_path):
    """from npz get samples"""
    print(f"📖 Loading samples from {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    num_samples = int(data['num_samples'])
    
    samples = []
    for i in range(num_samples):
        sample = data[f'sample_{i}'].item()  
        samples.append(sample)
    print(f"✅ Loaded {len(samples)} samples")
    return samples

def analyze_and_plot_samples(samples, save_dir='/home/goatoine/Documents/Lanyue/models/GNN/result/'):
    """
    Analyze y_cls and y_reg distribution of samples and generate distribution plots

    Args:
        samples: list of dicts, each containing keys 'y_cls' and 'y_reg'
        save_dir: directory to save plots
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # ===== 1. y_cls class distribution =====
    y_cls_values = [int(sample['y_cls']) for sample in samples]
    unique, counts = np.unique(y_cls_values, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"📈 y_cls class distribution: {class_dist}")

    # --- Plot y_cls histogram ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique, counts, color='skyblue', edgecolor='black', alpha=0.8)

    # Add value annotations on each bar
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    plt.xlabel('Class label (y_cls)')
    plt.ylabel('Count')
    plt.title('y_cls Class Distribution')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save class distribution plot
    save_path_cls = os.path.join(save_dir, 'y_cls_distribution.png')
    plt.savefig(save_path_cls, dpi=300, bbox_inches='tight')
    print(f"✅ y_cls distribution plot saved to: {save_path_cls}")
    plt.close()

    # ===== 2. y_reg statistics =====
    y_reg_values = [float(sample['y_reg']) for sample in samples]
    print(f"📊 y_reg statistics:")
    print(f"  Min:  {np.min(y_reg_values):.6f}")
    print(f"  Max:  {np.max(y_reg_values):.6f}")
    print(f"  Mean: {np.mean(y_reg_values):.6f}")
    print(f"  Std:  {np.std(y_reg_values):.6f}")

    # --- Plot y_reg histogram ---
    plt.figure(figsize=(10, 6))
    plt.hist(y_reg_values, bins=50, alpha=0.7, edgecolor='black', color='salmon')
    plt.xlabel('y_reg values')
    plt.ylabel('Frequency')
    plt.title('y_reg Distribution Histogram')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path_reg = os.path.join(save_dir, 'y_reg_distribution.png')
    plt.savefig(save_path_reg, dpi=300, bbox_inches='tight')
    print(f"✅ y_reg distribution plot saved to: {save_path_reg}")
    plt.close()



@register("reader", "opf_vA")   #
class OPFReaderVA:
    def __init__(self):
        self.samples = self._load_v1()  
    def _load_v1(self):
        sample_file = "/home/goatoine/Documents/Lanyue/data/data_for_GCN/data_basic_GCN/case2746wop_1029_samples.npz"
        samples = load_samples_from_npz(sample_file)
        print(f"📊 总计: {len(samples)} 个样本")
        samples = convert_samples_to_torch(samples)
        samples = [s for s in samples if "original_0" in s.get("scenario_id", "")]
        # keep only one sample for each loadfile
        seen_files = set()
        unique_samples = []
        for sample in samples:
            source_file = sample.get('source_file', '')
            if source_file not in seen_files:
                seen_files.add(source_file)
                unique_samples.append(sample)

        print(f"🔧 After deduplication: {len(unique_samples)} samples (from {len(samples)})")
        samples = unique_samples
        analyze_and_plot_samples(samples)
        # keep the features we need
        key_features = ["A_grid", "node_load", "y_cls", "y_arr_reg"]
        filtered_samples = []
        for sample in samples:
            filtered_sample = {}
            for key in key_features:
                if key in sample:
                    filtered_sample[key] = sample[key]
                else:
                    print(f"⚠️ Warning: {key} not found in sample")
            filtered_samples.append(filtered_sample)

        samples = filtered_samples
        print(f"🔧 After feature filtering: kept {key_features}")
       
        return samples
    def load(self):
        return self.samples


@register("reader", "opf_vB")   #
class OPFReaderVB:
    def __init__(self):
        self.samples = self._load_v1()  
    def _load_v1(self):
        sample_file = "/home/goatoine/Documents/Lanyue/data/data_for_GCN/data_basic_GCN/case2746wop_1029_samples.npz"
        samples = load_samples_from_npz(sample_file)
        print(f"📊 总计: {len(samples)} 个样本")
        samples = convert_samples_to_torch(samples)
        # keep only one sample for each loadfile
        seen_files = set()
        unique_samples = []
        for sample in samples:
            source_file = sample.get('source_file', '')
            if source_file not in seen_files:
                seen_files.add(source_file)
                unique_samples.append(sample)

        print(f"🔧 After deduplication: {len(unique_samples)} samples (from {len(samples)})")
        samples = unique_samples
        analyze_and_plot_samples(samples)
        # keep the features we need
        key_features = ["A_grid","A","node_load", "y_cls", "y_arr_reg"]
        filtered_samples = []
        for sample in samples:
            filtered_sample = {}
            for key in key_features:
                if key in sample:
                    filtered_sample[key] = sample[key]
                else:
                    print(f"⚠️ Warning: {key} not found in sample")
            filtered_samples.append(filtered_sample)

        samples = filtered_samples
        print(f"🔧 After feature filtering: kept {key_features}")
       
        return samples
    def load(self):
        return self.samples
# @register("reader", "opf_v2")   # 第二种版本（4500 samples）
# class OPFReaderV2(OPFReaderV1):
#     def _load_v1(self):  # 可以改名叫 _load_v2
#         # 这里是新的读取逻辑
#         # return [dict(...), dict(...), ...]   长度=4500
#         ...
#         return samples
