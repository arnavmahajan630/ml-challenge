"""
src/blocking/biencoder_train.py — Contrastive fine-tuning for B5 (OOF loop).

Trains a SentenceTransformer model using MultipleNegativesRankingLoss.
Batches contain (anchor, positive) pairs. 20% of pairs are label-free
augmentations (name, noised_name) to teach format invariance.

OOF loop: Train on N-1 folds, predict on fold N.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from ..data.augment import make_label_free_pairs


def _prepare_text(record: Dict[str, str]) -> str:
    n = str(record.get("name_clean", ""))
    a = str(record.get("addr_clean", ""))
    return f"{n} , {a}"


def train_biencoder_fold(
    train_s1_df: pd.DataFrame,
    train_s23_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    val_s1_df: pd.DataFrame,  # Used only for eval/saving, not training
    model_name: str,
    output_dir: str | Path,
    max_len: int = 96,
    batch_size: int = 64,
    epochs: int = 3,
    lr: float = 2e-5,
    label_free_frac: float = 0.20,
    seed: int = 42,
) -> str:
    """Train a bi-encoder for a single fold and return the saved model path.

    Returns:
        Path to the saved model directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Gather all GT positive pairs for training
    s1_dict = train_s1_df.set_index("entity_id").to_dict("index")
    s23_dict = train_s23_df.set_index("entity_id").to_dict("index")
    
    gt_dict = gt_df.set_index("source1_entity_id")["matched_ids_list"].to_dict()
    
    train_examples: List[InputExample] = []
    
    for s1_id, data in s1_dict.items():
        matched = gt_dict.get(s1_id, [])
        s1_text = _prepare_text(data)  # type: ignore
        for m_id in matched:
            if m_id in s23_dict:
                s23_text = _prepare_text(s23_dict[m_id])  # type: ignore
                train_examples.append(InputExample(texts=[s1_text, s23_text], label=1.0))
                
    # 2. Generate label-free pairs (augmentation)
    n_label_free = int(len(train_examples) * label_free_frac / max(1e-9, 1.0 - label_free_frac))
    all_train_records = list(s1_dict.values()) + list(s23_dict.values())
    
    # Needs abbrev map for augmentation? Not strictly passed, but augment handles None
    lf_pairs = make_label_free_pairs(all_train_records, n_pairs=n_label_free, p=0.3, seed=seed)
    
    for orig, noised in lf_pairs:
        t_orig = _prepare_text(orig)
        t_noised = _prepare_text(noised)
        train_examples.append(InputExample(texts=[t_orig, t_noised], label=1.0))
        
    random.seed(seed)
    random.shuffle(train_examples)
    
    # 3. Train model
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_len
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': lr},
        show_progress_bar=True,
        output_path=str(output_dir),
        save_best_model=False,  # Save at end
        use_amp=True
    )
    
    # Free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return str(output_dir)
