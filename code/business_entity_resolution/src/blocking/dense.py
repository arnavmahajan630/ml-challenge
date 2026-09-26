"""
src/blocking/dense.py — Dense retrieval blocking (B4, B5) using FAISS.

B4: Pretrained multilingual-e5-base.
B5: Fine-tuned bi-encoder (mdeberta or e5).

Embeds name_clean + " [SEP] " + addr_clean.
Uses FAISS IndexFlatIP (Inner Product) since embeddings are L2 normalized.
"""
from __future__ import annotations

import gc
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def embed_texts(
    texts: List[str],
    model_name: str,
    max_len: int = 96,
    batch_size: int = 256,
    device: str = "cuda",
) -> np.ndarray:
    """Embed texts using a HuggingFace sentence-transformers model.

    Args:
        texts: List of string texts to embed.
        model_name: HuggingFace model ID or local path.
        max_len: Max sequence length.
        batch_size: Inference batch size.
        device: 'cuda' or 'cpu'.

    Returns:
        np.ndarray of shape (len(texts), hidden_dim), L2-normalized.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Disable dropout/grad
    dataset = TextDataset(texts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embs = []
    with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16 if "cuda" in device else torch.float32):
        for batch in tqdm(loader, desc=f"Embedding {model_name}"):
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            
            outputs = model(**inputs)
            # Mean pooling (excluding pad tokens)
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embs = sum_embeddings / sum_mask
            
            # L2 normalize
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().numpy())

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.vstack(all_embs)


class DenseBlocker:
    """FAISS-based dense retrieval blocker."""

    def __init__(self, blocker_id: str, model_name: str, top_k: int = 30, max_len: int = 96, device: str = "cuda") -> None:
        self.blocker_id = blocker_id
        self.model_name = model_name
        self.top_k = top_k
        self.max_len = max_len
        self.device = device
        self._is_fit = False
        self._index = None
        self._index_ids = []

    def _prepare_texts(self, df: pd.DataFrame) -> List[str]:
        # Prefix with 'query: ' for e5 models if applicable, though typically symmetric is fine for this task.
        # We will use name_clean [SEP] addr_clean. If [SEP] isn't supported, we use " , ".
        # For e5/mdeberta, " " or " , " is safer without tokenizer specifics.
        # Spec implies using name_clean and addr_clean directly.
        names = df["name_clean"].fillna("").astype(str).tolist()
        addrs = df["addr_clean"].fillna("").astype(str).tolist()
        return [f"{n} , {a}" for n, a in zip(names, addrs)]

    def retrieve(
        self,
        s1_df: pd.DataFrame,
        s23_df: pd.DataFrame,
    ) -> pd.DataFrame:
        import faiss

        s1_texts = self._prepare_texts(s1_df)
        s23_texts = self._prepare_texts(s23_df)

        print(f"[{self.blocker_id}] Embedding S1 queries...")
        s1_embs = embed_texts(s1_texts, self.model_name, self.max_len, device=self.device)
        
        print(f"[{self.blocker_id}] Embedding S2/S3 index...")
        s23_embs = embed_texts(s23_texts, self.model_name, self.max_len, device=self.device)

        dim = s23_embs.shape[1]
        res = faiss.StandardGpuResources() if "cuda" in self.device and faiss.get_num_gpus() > 0 else None
        
        index = faiss.IndexFlatIP(dim)
        if res:
            index = faiss.index_cpu_to_gpu(res, 0, index)
            
        index.add(s23_embs)
        
        # S1 -> S2/S3
        scores_fwd, indices_fwd = index.search(s1_embs, min(self.top_k, len(s23_texts)))
        
        s1_ids = s1_df["entity_id"].tolist()
        s23_ids = s23_df["entity_id"].tolist()
        
        rows = []
        for i, (idxs, scs) in enumerate(zip(indices_fwd, scores_fwd)):
            s1_id = s1_ids[i]
            for rank, (idx, score) in enumerate(zip(idxs, scs)):
                if idx != -1:
                    rows.append({
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": s23_ids[idx],
                        f"{self.blocker_id.lower()}_score": float(score),
                        f"{self.blocker_id.lower()}_rank": rank + 1
                    })
                    
        fwd_df = pd.DataFrame(rows)
        
        # S2/S3 -> S1 (Reverse)
        # To avoid rebuilding index, we build S1 index and query with S2/S3
        del index
        if res:
            gc.collect()
            
        index_rev = faiss.IndexFlatIP(dim)
        if res:
            index_rev = faiss.index_cpu_to_gpu(res, 0, index_rev)
            
        index_rev.add(s1_embs)
        scores_rev, indices_rev = index_rev.search(s23_embs, min(self.top_k, len(s1_texts)))
        
        rev_rows = []
        for i, (idxs, scs) in enumerate(zip(indices_rev, scores_rev)):
            s23_id = s23_ids[i]
            for rank, (idx, score) in enumerate(zip(idxs, scs)):
                if idx != -1:
                    rev_rows.append({
                        "source1_entity_id": s1_ids[idx],
                        "candidate_entity_id": s23_id,
                        f"{self.blocker_id.lower()}_rev_score": float(score),
                        f"{self.blocker_id.lower()}_rev_rank": rank + 1
                    })
                    
        rev_df = pd.DataFrame(rev_rows)
        
        merged = pd.merge(fwd_df, rev_df, on=["source1_entity_id", "candidate_entity_id"], how="outer")
        score_cols = [c for c in merged.columns if "score" in c]
        rank_cols = [c for c in merged.columns if "rank" in c]
        merged[score_cols] = merged[score_cols].fillna(0.0)
        merged[rank_cols] = merged[rank_cols].fillna(999)
        
        return merged
