"""
src/models/cross_encoder.py — CrossEncoder inference and fine-tuning.

Uses HuggingFace sentence-transformers CrossEncoder.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def _prepare_ce_text(record: Dict[str, str]) -> str:
    n = str(record.get("name_clean", ""))
    a = str(record.get("addr_clean", ""))
    return f"{n} , {a}"


def train_cross_encoder_fold(
    train_pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    model_name: str,
    output_dir: str,
    max_len: int = 160,
    batch_size: int = 32,
    epochs: int = 3,
    lr: float = 2e-5,
    neg_pos_ratio: int = 6,
) -> str:
    """Train CE on given pairs.
    Pairs DataFrame should have: source1_entity_id, candidate_entity_id, label.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    s1_dict = s1_df.set_index("entity_id").to_dict("index")
    s23_dict = s23_df.set_index("entity_id").to_dict("index")
    
    # Subsample negatives to hit neg_pos_ratio
    pos_mask = train_pairs["label"] == 1
    pos_df = train_pairs[pos_mask]
    neg_df = train_pairs[~pos_mask]
    
    target_neg = len(pos_df) * neg_pos_ratio
    if len(neg_df) > target_neg:
        neg_df = neg_df.sample(n=target_neg, random_state=42)
        
    train_data = pd.concat([pos_df, neg_df], ignore_index=True)
    
    examples = []
    for _, row in train_data.iterrows():
        s1 = row["source1_entity_id"]
        c = row["candidate_entity_id"]
        lbl = float(row["label"])
        
        t1 = _prepare_ce_text(s1_dict.get(s1, {}))
        t2 = _prepare_ce_text(s23_dict.get(c, {}))
        
        examples.append(InputExample(texts=[t1, t2], label=lbl))
        
    random.shuffle(examples)
    
    model = CrossEncoder(model_name, num_labels=1, max_length=max_len)
    
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    
    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': lr},
        output_path=output_dir,
        use_amp=True
    )
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return output_dir


def predict_cross_encoder(
    pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    model_path: str,
    max_len: int = 160,
    batch_size: int = 128
) -> np.ndarray:
    """Predict probabilities for pairs using CE."""
    s1_dict = s1_df.set_index("entity_id").to_dict("index")
    s23_dict = s23_df.set_index("entity_id").to_dict("index")
    
    texts = []
    for _, row in pairs_df.iterrows():
        s1 = row["source1_entity_id"]
        c = row["candidate_entity_id"]
        t1 = _prepare_ce_text(s1_dict.get(s1, {}))
        t2 = _prepare_ce_text(s23_dict.get(c, {}))
        texts.append([t1, t2])
        
    model = CrossEncoder(model_path, max_length=max_len)
    
    preds = model.predict(texts, batch_size=batch_size, show_progress_bar=True, apply_softmax=False)
    
    # Apply sigmoid if model outputs logits
    from scipy.special import expit
    probs = expit(preds)
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return probs
