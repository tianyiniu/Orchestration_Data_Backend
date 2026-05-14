from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "microsoft/harrier-oss-v1-0.6b"
DEFAULT_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
MAX_LENGTH = 512


def format_query(query: str, task: str = DEFAULT_TASK) -> str:
    return f"Instruct: {task}\nQuery:{query}"


def last_token_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    left_padded = (attn_mask[:, -1].sum() == attn_mask.shape[0])
    if left_padded:
        return last_hidden[:, -1]
    seq_lens = attn_mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.size(0)), seq_lens]


class HarrierEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        max_length: int = MAX_LENGTH,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            cache_dir=cache_dir,
        )
        self.model = (
            AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
            )
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def _embed(self, texts: List[str]) -> np.ndarray:
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**tok)
        emb = last_token_pool(out.last_hidden_state, tok["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.float().cpu().numpy()

    def encode_query(self, query: str) -> np.ndarray:
        return self._embed([format_query(query)])

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        return self._embed([format_query(q) for q in queries])

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        return self._embed(texts)
