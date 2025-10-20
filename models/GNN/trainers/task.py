import torch
from registries import register
from registries import build  
import evaluation.evaluators 
@register("task", "opf_basic_task")
class OPFBasicTask:
    def __init__(self, model, loss_fn, device="cpu", lr=1e-3):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)

    def _to_device(self, batch):
        for k, v in batch.items():
            if torch.is_tensor(v): batch[k] = v.to(self.device)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if torch.is_tensor(vv): v[kk] = vv.to(self.device)
        return batch

    def train_one_epoch(self, loader):
        self.model.train()
        total = 0.0
        for batch in loader:
            batch = self._to_device(batch)
            self.opt.zero_grad()
            out = self.model(batch)
            loss = self.loss_fn(out, batch)
            loss.backward()
            self.opt.step()
            total += loss.item()
        return total / max(1, len(loader))

    @torch.no_grad()
    def validate(self, loader):  # 验证损失
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            batch = self._to_device(batch)
            out = self.model(batch)
            loss = self.loss_fn(out, batch)  # 使用相同的损失函数
            total_loss += loss.item()
        return total_loss / max(1, len(loader))
    
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
        evaluator = build("evaluator", name, output_dir=output_dir)
        return evaluator.run(self.model, loader, device=self.device, output_dir=output_dir)
