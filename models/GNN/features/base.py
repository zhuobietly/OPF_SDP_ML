from abc import ABC, abstractmethod
import torch

class FeaturePipeline(ABC):
    @abstractmethod
    def node_features(self, graph_dict, raw) -> torch.Tensor:
        raise NotImplementedError

    def edge_features(self, graph_dict, raw):
        return graph_dict.get("edge_attr")

    def finalize(self, X, E):
        return X, E
