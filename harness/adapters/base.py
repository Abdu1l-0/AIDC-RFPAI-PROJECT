from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseModelAdapter(ABC):
    def __init__(self, model_key: str, endpoint_url: str, model_name: str):
        self.model_key = model_key
        self.endpoint_url = endpoint_url
        self.model_name = model_name

    @abstractmethod
    def invoke(self, messages: list, decoding_params: Dict[str, Any]) -> Dict[str, Any]:
        pass
