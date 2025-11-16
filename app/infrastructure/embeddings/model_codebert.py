import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class CodeEmbeddingModel:
    def __init__(self, model_name="microsoft/codebert-base", device=None):
        self.device = self.pick_device() if device is None else torch.device(device)
        print(f"Using device: {self.device}")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @staticmethod
    def pick_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Metal backend
        if torch.cuda.is_available():
            return torch.device("cuda")  # NVIDIA GPU
        return torch.device("cpu")  # fallback

    @staticmethod
    def _l2(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x) + 1e-12
        return (x / n).astype("float32")

    def encode(self, text: str) -> np.ndarray:
        with torch.inference_mode():
            toks = self.tok(text, return_tensors="pt", truncation=True, max_length=512)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            out = self.model(**toks).last_hidden_state  # [1, L, H]
            mask = toks["attention_mask"].unsqueeze(-1)  # [1, L, 1]
            mean = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vec = mean.squeeze(0).detach().cpu().numpy()
        return self._l2(vec)
