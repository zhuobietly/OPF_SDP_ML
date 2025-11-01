import os
import csv
from datetime import datetime
import torch

class ModelLogger:
    def __init__(self, model, optimizer=None, log_dir="./logs"):
        self.model = model
        self.optimizer = optimizer
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "grad_log.csv")
        self._pre_params = {}
        self._logged_model_info = False

    def record_model(self):
        
        if not self._logged_model_info:
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            model_text = str(self.model).replace("\n", "; ")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([])
                writer.writerow(["==== MODEL SUMMARY ====", "", "", "", "", ""])
                writer.writerow(["time", timestamp, "", "", "", ""])
                writer.writerow(["model_structure", model_text, "", "", "", ""])
                writer.writerow(["total_params", total, "", "", "", ""])
                writer.writerow(["trainable_params", trainable, "", "", "", ""])
                writer.writerow(["======================", "", "", "", "", ""])
                writer.writerow([])

            self._logged_model_info = True

    def pre_step(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "name", "grad_norm", "update_norm", "param_norm", "lr"])
        self._pre_params = {
            n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad
        }

    def log_gradients(self, step: int):
        lr = None
        if self.optimizer is not None:
            try:
                lr = self.optimizer.param_groups[0].get("lr", None)
            except Exception:
                pass

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                grad_norm = p.grad.norm().item() if p.grad is not None else float("nan")
                update_norm = float("nan")
                if name in self._pre_params:
                    update_norm = (p.detach() - self._pre_params[name]).norm().item()
                writer.writerow([step, name, grad_norm, update_norm, p.detach().norm().item(), lr])
