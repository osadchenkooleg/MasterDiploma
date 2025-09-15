import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class CodeEmbeddingModel:
    def __init__(
        self, model_name: str = "microsoft/codebert-base", device: str | None = None
    ):
        self.device = self.pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x) + 1e-12
        return x / n

    def encode(self, text: str) -> np.ndarray:
        with torch.inference_mode():
            toks = self.tok(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            out = self.model(**toks).last_hidden_state[:, 0, :]  # CLS pooling
            vec = out.detach().cpu().numpy().astype("float32").squeeze(0)
        return self._l2_normalize(vec)

    def pick_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
