from abc import ABC, abstractmethod

class GraphBuilder(ABC):
    @abstractmethod
    def build(self, raw):
        raise NotImplementedError
