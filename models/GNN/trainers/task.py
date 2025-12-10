import os, sys, yaml, torch
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
import matplotlib.pyplot as plt
import numpy as np
from registries import register
from registries import build  
import evaluation.evaluators 
from gcn_utils.model_logger import ModelLogger
from torch.optim import AdamW


@register("task", "opf_basicA_task")
class OPFBasicTask:
    def __init__(self, model, loss_fn, device="cpu", lr=1e-3, log_dir=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        
        self.train_losses = []
        self.val_losses = []
        self.log_dir = log_dir
        
        if log_dir:
            self.logger = ModelLogger(self.model, self.opt, log_dir=log_dir)
            self.logger.record_model()
        else:
            self.logger = None

    def _to_device(self, batch):
        for k, v in batch.items():
            if torch.is_tensor(v): batch[k] = v.to(self.device)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if torch.is_tensor(vv): v[kk] = vv.to(self.device)
        return batch

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        total = 0.0
        for step, batch in enumerate(loader):
            batch = self._to_device(batch)
            self.opt.zero_grad()
            out = self.model(batch, return_all_H=True)
            H_list = out["H_list"]
            for l, H in enumerate(H_list):
                # H shape = (B, N, F)
                std = H.std(dim=1).mean().item()  # 每个节点的std平均
                print(f"Layer {l}: mean node feature std = {std:.6f}")
            loss = self.loss_fn(out, batch)
            loss.backward()
            if self.logger:
                self.logger.pre_step()
            
            self.opt.step()
            
            if self.logger:
                self.logger.log_gradients(epoch, step)
                
            total += loss.item()
        
        avg_loss = total / max(1, len(loader))
        self.train_losses.append(avg_loss)  # 记录训练损失
        return avg_loss
    
    @torch.no_grad()
    def validate(self, loader, epoch):  # 验证损失
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            batch = self._to_device(batch)
            out = self.model(batch)
            loss = self.loss_fn(out, batch)  # 使用相同的损失函数
            total_loss += loss.item()
        
        avg_loss = total_loss / max(1, len(loader))
        self.val_losses.append(avg_loss)  # 记录验证损失
        return avg_loss

    def plot_loss_curves(self, cfg):
        save_path = cfg["evaluator"]["output_dir"]
        """绘制训练和验证损失曲线"""
        if not self.train_losses:
            print("No training losses to plot")
            return
            
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 绘制训练损失
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        
        # 绘制验证损失（如果有）
        if self.val_losses:
            val_epochs = range(1, len(self.val_losses) + 1)
            plt.plot(val_epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加数值标注（可选，只在epoch较少时显示）
        if len(self.train_losses) <= 20:
            for i, loss in enumerate(self.train_losses):
                plt.annotate(f'{loss:.3f}', (i+1, loss), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
        
        plt.tight_layout()
        

        save_path = os.path.join(save_path, "loss_curves.png")

        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Loss curves saved to: {save_path}")
        
        # 打印损失统计
        print(f"\n📊 Loss Statistics:")
        print(f"Final Training Loss: {self.train_losses[-1]:.6f}")
        if self.val_losses:
            print(f"Final Validation Loss: {self.val_losses[-1]:.6f}")
            print(f"Best Validation Loss: {min(self.val_losses):.6f} (Epoch {self.val_losses.index(min(self.val_losses))+1})")

    def save_loss_data(self, cfg):
        save_path = cfg["evaluator"]["output_dir"]
        save_path = os.path.join(save_path, "loss_data.csv")
        import csv
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss'])
            
            max_len = max(len(self.train_losses), len(self.val_losses))
            for i in range(max_len):
                train_loss = self.train_losses[i] if i < len(self.train_losses) else ''
                val_loss = self.val_losses[i] if i < len(self.val_losses) else ''
                writer.writerow([i+1, train_loss, val_loss])
        
        print(f"Loss data saved to: {save_path}")
    
    @torch.no_grad()
    def test_with_evaluator(self, loader, cfg):
        """
        cfg 示例 1：一维标量回归
        cfg = {"evaluator": {"name": "scalar_only"}, "output_dir": "./outputs"}

        cfg 示例 2：数组回归 + 分类
        cfg = {"evaluator": {"name": "arr_cls"}, "output_dir": "./outputs"}
        """
        name = cfg["evaluator"]["name"]
        output_dir = cfg["evaluator"]["output_dir"]
        num_classes = cfg["model"].get("out_array_dim", 15)
        evaluator = build("evaluator", name, output_dir=output_dir, num_classes=num_classes)
        
        return evaluator.run(self.model, loader, device=self.device, output_dir=output_dir)

@register("task", "opf_basicA_task_Gsmooth")
class OPFBasicTask:
    def __init__(self, model, loss_fn, device="cpu", lr=1e-3, log_dir=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        ###########################################
        # 🔥🔥🔥 在这里冻结 head_array（如果有）
        ###########################################
        if hasattr(self.model, "head_array") and self.model.head_array is not None:
            print("🔒 Freezing head_array parameters...")
            for p in self.model.head_array.parameters():
                p.requires_grad = False
        ###########################################
        # ❗ 注意：optimizer 要放在 freeze 之后
        self.opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        self.train_losses = []
        self.val_losses = []
        self.Gsmooth_stds = []
        self.log_dir = log_dir
        
        if log_dir:
            self.logger = ModelLogger(self.model, self.opt, log_dir=log_dir)
            self.logger.record_model()
        else:
            self.logger = None

    def _to_device(self, batch):
        for k, v in batch.items():
            if torch.is_tensor(v): batch[k] = v.to(self.device)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if torch.is_tensor(vv): v[kk] = vv.to(self.device)
        return batch

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        total = 0.0
        
        for step, batch in enumerate(loader):
            batch = self._to_device(batch)
            self.opt.zero_grad()
            out = self.model(batch, return_all_H=True)
            H_list = out["H_list"]
            std_list = []
            for l, H in enumerate(H_list):
                # H shape = (B, N, F)
                std = H.std(dim=1).mean().item()  # 每个节点的std平均
                print(f"Layer {l}: mean node feature std = {std:.6f}")
                std_list.append(std)
            
            self.Gsmooth_stds.append(std_list)
            loss = self.loss_fn(out, batch)
            loss.backward()
            if self.logger:
                self.logger.pre_step()
            
            self.opt.step()
            
            if self.logger:
                self.logger.log_gradients(epoch, step)
                
            total += loss.item()
        
        avg_loss = total / max(1, len(loader))
        self.train_losses.append(avg_loss)  # 记录训练损失
        return avg_loss
    
    @torch.no_grad()
    def validate(self, loader, epoch):  # 验证损失
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            batch = self._to_device(batch)
            out = self.model(batch)
            loss = self.loss_fn(out, batch)  # 使用相同的损失函数
            total_loss += loss.item()
        
        avg_loss = total_loss / max(1, len(loader))
        self.val_losses.append(avg_loss)  # 记录验证损失
        return avg_loss

    def plot_loss_curves(self, cfg):
        save_path = cfg["evaluator"]["output_dir"]
        """绘制训练和验证损失曲线"""
        if not self.train_losses:
            print("No training losses to plot")
            return
            
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 绘制训练损失
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        
        # 绘制验证损失（如果有）
        if self.val_losses:
            val_epochs = range(1, len(self.val_losses) + 1)
            plt.plot(val_epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加数值标注（可选，只在epoch较少时显示）
        if len(self.train_losses) <= 20:
            for i, loss in enumerate(self.train_losses):
                plt.annotate(f'{loss:.3f}', (i+1, loss), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
        
        plt.tight_layout()
        

        save_path = os.path.join(save_path, "loss_curves.png")

        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Loss curves saved to: {save_path}")
        
        # 打印损失统计
        print(f"\n📊 Loss Statistics:")
        print(f"Final Training Loss: {self.train_losses[-1]:.6f}")
        if self.val_losses:
            print(f"Final Validation Loss: {self.val_losses[-1]:.6f}")
            print(f"Best Validation Loss: {min(self.val_losses):.6f} (Epoch {self.val_losses.index(min(self.val_losses))+1})")

    def save_loss_data(self, cfg):
        save_path = cfg["evaluator"]["output_dir"]
        save_path = os.path.join(save_path, "loss_data.csv")
        import csv
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Gsmooth_Stds'])
            
            max_len = max(len(self.train_losses), len(self.val_losses))
            for i in range(max_len):
                train_loss = self.train_losses[i] if i < len(self.train_losses) else ''
                val_loss = self.val_losses[i] if i < len(self.val_losses) else ''
                gsmooth_stds = self.Gsmooth_stds[i] if i < len(self.Gsmooth_stds) else ''
                writer.writerow([i+1, train_loss, val_loss, gsmooth_stds])
        
        print(f"Loss data saved to: {save_path}")
    
    @torch.no_grad()
    def test_with_evaluator(self, loader, cfg):
        """
        cfg 示例 1：一维标量回归
        cfg = {"evaluator": {"name": "scalar_only"}, "output_dir": "./outputs"}

        cfg 示例 2：数组回归 + 分类
        cfg = {"evaluator": {"name": "arr_cls"}, "output_dir": "./outputs"}
        """
        name = cfg["evaluator"]["name"]
        output_dir = cfg["evaluator"]["output_dir"]
        num_classes = cfg["model"].get("out_array_dim", 15)
        evaluator = build("evaluator", name, output_dir=output_dir, num_classes=num_classes)
        
        return evaluator.run(self.model, loader, device=self.device, output_dir=output_dir)




@register("task", "opf_basicA_dirl_task")
class OPFBasicTask:
        def __init__(self, model, loss_fn, device="cpu", lr=1e-3, log_dir=None):
            self.model = model.to(device)
            self.loss_fn = loss_fn
            self.device = device
            gcn_backbone_params = []
            head_params = []

            for name, p in model.named_parameters():
                if "gcns" in name:
                    gcn_backbone_params.append(p)
                else:
                    head_params.append(p)

            backbone_lr = 3 * lr  
            head_lr     = 1 * lr   

            self.opt = AdamW(
                [
                    {"params": gcn_backbone_params, "lr": backbone_lr},
                    {"params": head_params,        "lr": head_lr},
                ],
                weight_decay=1e-4,
            )
            
            # 添加损失记录列表
            self.train_losses = []
            self.val_losses = []
            self.log_dir = log_dir
            
            if log_dir:
                self.logger = ModelLogger(self.model, self.opt, log_dir=log_dir)
                self.logger.record_model()
            else:
                self.logger = None

        def _to_device(self, batch):
            for k, v in batch.items():
                if torch.is_tensor(v): batch[k] = v.to(self.device)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if torch.is_tensor(vv): v[kk] = vv.to(self.device)
            return batch

        def train_one_epoch(self, loader, epoch):
            self.model.train()
            total = 0.0
            for step, batch in enumerate(loader):
                batch = self._to_device(batch)
                self.opt.zero_grad()
                out = self.model(batch)
                loss = self.loss_fn(out, batch)
                loss.backward()
                
                if self.logger:
                    self.logger.pre_step()
                
                self.opt.step()
                
                if self.logger:
                    self.logger.log_gradients(epoch,step)
                    
                total += loss.item()
            
            avg_loss = total / max(1, len(loader))
            self.train_losses.append(avg_loss)  # 记录训练损失
            return avg_loss

        @torch.no_grad()
        def validate(self, loader, epoch):  # 验证损失
            self.model.eval()
            total_loss = 0.0
            for batch in loader:
                batch = self._to_device(batch)
                out = self.model(batch)
                loss = self.loss_fn(out, batch)  # 使用相同的损失函数
                total_loss += loss.item()
            
            avg_loss = total_loss / max(1, len(loader))
            self.val_losses.append(avg_loss)  # 记录验证损失
            return avg_loss

        def plot_loss_curves(self, cfg):
            save_path = cfg["evaluator"]["output_dir"]
            """绘制训练和验证损失曲线"""
            if not self.train_losses:
                print("No training losses to plot")
                return
                
            plt.figure(figsize=(10, 6))
            
            epochs = range(1, len(self.train_losses) + 1)
            
            # 绘制训练损失
            plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
            
            # 绘制验证损失（如果有）
            if self.val_losses:
                val_epochs = range(1, len(self.val_losses) + 1)
                plt.plot(val_epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 添加数值标注（可选，只在epoch较少时显示）
            if len(self.train_losses) <= 20:
                for i, loss in enumerate(self.train_losses):
                    plt.annotate(f'{loss:.3f}', (i+1, loss), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=8)
            
            plt.tight_layout()
            

            save_path = os.path.join(save_path, "loss_curves.png")

            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Loss curves saved to: {save_path}")
            
            # 打印损失统计
            print(f"\n📊 Loss Statistics:")
            print(f"Final Training Loss: {self.train_losses[-1]:.6f}")
            if self.val_losses:
                print(f"Final Validation Loss: {self.val_losses[-1]:.6f}")
                print(f"Best Validation Loss: {min(self.val_losses):.6f} (Epoch {self.val_losses.index(min(self.val_losses))+1})")

        def save_loss_data(self, cfg):
            save_path = cfg["evaluator"]["output_dir"]
            save_path = os.path.join(save_path, "loss_data.csv")
            import csv
            with open(save_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss'])
                
                max_len = max(len(self.train_losses), len(self.val_losses))
                for i in range(max_len):
                    train_loss = self.train_losses[i] if i < len(self.train_losses) else ''
                    val_loss = self.val_losses[i] if i < len(self.val_losses) else ''
                    writer.writerow([i+1, train_loss, val_loss])
            
            print(f"Loss data saved to: {save_path}")
        
        @torch.no_grad()
        def test_with_evaluator(self, loader, cfg):
            """
            cfg 示例 1：一维标量回归
            cfg = {"evaluator": {"name": "scalar_only"}, "output_dir": "./outputs"}

            cfg 示例 2：数组回归 + 分类
            cfg = {"evaluator": {"name": "arr_cls"}, "output_dir": "./outputs"}
            """
            name = cfg["evaluator"]["name"]
            output_dir = cfg["evaluator"]["output_dir"]
            num_classes = cfg["model"].get("out_array_dim", 15)
            evaluator = build("evaluator", name, output_dir=output_dir, num_classes=num_classes)
            
            return evaluator.run(self.model, loader, device=self.device, output_dir=output_dir)


        
@register("task", "opf_globalB_task")
class OPFBasicTask:
    def __init__(self, model, loss_fn, device="cpu", lr=1e-3, log_dir=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        
        # 添加损失记录列表
        self.train_losses = []
        self.val_losses = []
        self.log_dir = log_dir
        
        if log_dir:
            self.logger = ModelLogger(self.model, self.opt, log_dir=log_dir)
            self.logger.record_model()
        else:
            self.logger = None

    def _to_device(self, batch):
        for k, v in batch.items():
            if torch.is_tensor(v): batch[k] = v.to(self.device)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if torch.is_tensor(vv): v[kk] = vv.to(self.device)
        return batch

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        total = 0.0
        for step, batch in enumerate(loader):
            batch = self._to_device(batch)
            self.opt.zero_grad()
            out = self.model(batch)
            loss = self.loss_fn(out, batch)
            loss.backward()
            
            if self.logger:
                self.logger.pre_step()
            
            self.opt.step()
            
            if self.logger:
                self.logger.log_gradients(epoch, step)
                
            total += loss.item()
        
        avg_loss = total / max(1, len(loader))
        self.train_losses.append(avg_loss)  # 记录训练损失
        return avg_loss

    @torch.no_grad()
    def validate(self, loader, epoch):  # 验证损失
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            batch = self._to_device(batch)
            out = self.model(batch)
            loss = self.loss_fn(out, batch)  # 使用相同的损失函数
            total_loss += loss.item()
        
        avg_loss = total_loss / max(1, len(loader))
        self.val_losses.append(avg_loss)  # 记录验证损失
        return avg_loss

    def plot_loss_curves(self, cfg):
        save_path = cfg["evaluator"]["output_dir"]
        """绘制训练和验证损失曲线"""
        if not self.train_losses:
            print("No training losses to plot")
            return
            
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 绘制训练损失
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        
        # 绘制验证损失（如果有）
        if self.val_losses:
            val_epochs = range(1, len(self.val_losses) + 1)
            plt.plot(val_epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加数值标注（可选，只在epoch较少时显示）
        if len(self.train_losses) <= 20:
            for i, loss in enumerate(self.train_losses):
                plt.annotate(f'{loss:.3f}', (i+1, loss), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
        
        plt.tight_layout()
        

        save_path = os.path.join(save_path, "loss_curves.png")

        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Loss curves saved to: {save_path}")
        
        # 打印损失统计
        print(f"\n📊 Loss Statistics:")
        print(f"Final Training Loss: {self.train_losses[-1]:.6f}")
        if self.val_losses:
            print(f"Final Validation Loss: {self.val_losses[-1]:.6f}")
            print(f"Best Validation Loss: {min(self.val_losses):.6f} (Epoch {self.val_losses.index(min(self.val_losses))+1})")

    def save_loss_data(self, cfg):
        save_path = cfg["evaluator"]["output_dir"]
        save_path = os.path.join(save_path, "loss_data.csv")
        import csv
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss'])
            
            max_len = max(len(self.train_losses), len(self.val_losses))
            for i in range(max_len):
                train_loss = self.train_losses[i] if i < len(self.train_losses) else ''
                val_loss = self.val_losses[i] if i < len(self.val_losses) else ''
                writer.writerow([i+1, train_loss, val_loss])
        
        print(f"Loss data saved to: {save_path}")
    
    @torch.no_grad()
    def test_with_evaluator(self, loader, cfg):
        """
        cfg 示例 1：一维标量回归
        cfg = {"evaluator": {"name": "scalar_only"}, "output_dir": "./outputs"}

        cfg 示例 2：数组回归 + 分类
        cfg = {"evaluator": {"name": "arr_cls"}, "output_dir": "./outputs"}
        """
        name = cfg["evaluator"]["name"]
        output_dir = cfg["evaluator"]["output_dir"]
        num_classes = cfg["model"].get("out_array_dim", 15)
        evaluator = build("evaluator", name, output_dir=output_dir, num_classes=num_classes)
        return evaluator.run(self.model, loader, device=self.device, output_dir=output_dir)

