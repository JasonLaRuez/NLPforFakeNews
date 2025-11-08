# ============================================================
# Pipeline: Clean -> Split -> Back-Translate -> Sanitize
#           -> (optional) MLM -> Sanitize
#           -> EDA -> Sanitize  ===> train_final_eda
# ============================================================

# --------------------------- Imports ---------------------------
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Sequence

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

import torch
from transformers import (
    MarianMTModel, MarianTokenizer,
    AutoTokenizer, AutoModelForMaskedLM
)
from functools import lru_cache

from nltk.corpus import wordnet as wn

# ---------------------- Global Defaults -----------------------
TEXT_COL  = "statement"
LABEL_COL = "verdict"

# --------------------- Text Normalization ---------------------
def normalize_text(s: str) -> str:
    """Lowercase, collapse whitespace, normalize quotes."""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[‘’]", "'", s)
    return s

# ------------------ Exact/Near-Duplicate Ops ------------------
def drop_exact_dups(df: pd.DataFrame, text_col: str = TEXT_COL) -> pd.DataFrame:
    """Drop exact duplicates by normalized text."""
    out = df.copy()
    out["_norm"] = out[text_col].map(normalize_text)
    out = out.drop_duplicates(subset="_norm").drop(columns=["_norm"]).reset_index(drop=True)
    return out

def drop_near_dups_tfidf_radius(
    df: pd.DataFrame,
    text_col: str = TEXT_COL,
    ngram_range=(1, 2),
    min_df=2,
    sim_threshold=0.95,
    batch_size=4096,
) -> pd.DataFrame:
    """Remove near duplicates: cosine > sim_threshold (keep earliest occurrence)."""
    df = df.copy()
    texts = df[text_col].map(normalize_text).tolist()

    vec = TfidfVectorizer(stop_words="english", ngram_range=ngram_range, min_df=min_df)
    X = vec.fit_transform(texts)

    # cosine_sim > t  <=>  cosine_dist < 1 - t
    radius = max(1.0 - sim_threshold, 1e-9)
    nn = NearestNeighbors(metric="cosine", algorithm="brute", radius=radius).fit(X)

    n = X.shape[0]
    to_drop = np.zeros(n, dtype=bool)
    visited = np.zeros(n, dtype=bool)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        _, indices = nn.radius_neighbors(X[start:end], radius=radius, return_distance=True)
        for off, neigh_idx in enumerate(indices):
            i = start + off
            if visited[i] or to_drop[i]:
                continue
            for j in neigh_idx:
                if j != i and j > i:
                    to_drop[j] = True
            visited[i] = True

    return df.loc[~to_drop].reset_index(drop=True)

def clean_statement_duplicates(
    df: pd.DataFrame,
    text_col: str = TEXT_COL,
    sim_threshold: float = 0.95
) -> pd.DataFrame:
    """Global cleaning: exact + near duplicates (run ONCE before split)."""
    df1 = drop_exact_dups(df, text_col=text_col)
    df2 = drop_near_dups_tfidf_radius(df1, text_col=text_col, sim_threshold=sim_threshold)
    return df2

# ------------------------ Stratified Split ---------------------
def stratified_split(
    df: pd.DataFrame,
    label_col: str = LABEL_COL,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """3-way stratified split."""
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0 (got {total})")

    temp_frac = val_frac + test_frac
    train_df, temp_df = train_test_split(
        df, test_size=temp_frac, stratify=df[label_col], random_state=random_state
    )
    test_size_rel = test_frac / temp_frac
    val_df, test_df = train_test_split(
        temp_df, test_size=test_size_rel, stratify=temp_df[label_col], random_state=random_state
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

# --------------- Back-Translation (MarianMT) ------------------
@lru_cache(maxsize=16)
def _load_pair(src_lang: str, mid_lang: str):
    en2mid = f"Helsinki-NLP/opus-mt-{src_lang}-{mid_lang}"
    mid2en = f"Helsinki-NLP/opus-mt-{mid_lang}-{src_lang}"
    en2mid_tok = MarianTokenizer.from_pretrained(en2mid)
    en2mid_model = MarianMTModel.from_pretrained(en2mid)
    mid2en_tok = MarianTokenizer.from_pretrained(mid2en)
    mid2en_model = MarianMTModel.from_pretrained(mid2en)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    en2mid_model.to(device).eval()
    mid2en_model.to(device).eval()
    return en2mid_tok, en2mid_model, mid2en_tok, mid2en_model, device

@torch.inference_mode()
def back_translate_batch(
    texts: List[str],
    mid_lang: str = "fr",
    src_lang: str = "en",
    batch_size: int = 16,
    max_length: int = 256,
    do_sample: bool = False,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 1.0,
) -> List[str]:
    en2mid_tok, en2mid_model, mid2en_tok, mid2en_model, device = _load_pair(src_lang, mid_lang)
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        # EN -> MID
        enc = en2mid_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        mid_ids = en2mid_model.generate(
            **enc, max_length=max_length, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature
        )
        mid_txt = en2mid_tok.batch_decode(mid_ids, skip_special_tokens=True)

        # MID -> EN
        enc2 = mid2en_tok(mid_txt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        en_ids = mid2en_model.generate(
            **enc2, max_length=max_length, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature
        )
        out.extend(mid2en_tok.batch_decode(en_ids, skip_special_tokens=True))
    return out

def backtranslate_balance_expand_marian(
    df: pd.DataFrame,
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
    target_langs: Union[str, Sequence[str]] = ("fr",),
    factor: float = 0.0,
    random_state: int = 42,
    batch_size: int = 16,
    max_length: int = 256,
    do_sample: bool = False,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 1.0,
    verbose: bool = True
) -> pd.DataFrame:
    """Balance classes via BT; optionally expand each class by `factor`."""
    rng = np.random.RandomState(random_state)
    if isinstance(target_langs, str):
        target_langs = [target_langs]

    class_counts = df[label_col].value_counts()
    max_count = class_counts.max()
    if verbose:
        print("Initial class counts:\n", class_counts, "\n")

    new_rows = []
    # Balance
    for label, count in class_counts.items():
        if count < max_count:
            n_needed = max_count - count
            subset = df[df[label_col] == label]
            to_bt = subset.sample(n=n_needed, replace=True, random_state=random_state)[text_col].tolist()
            pivot = rng.choice(target_langs)
            if verbose:
                print(f"Balancing '{label}': +{n_needed} via {pivot.upper()}")
            bt_texts = back_translate_batch(
                to_bt, mid_lang=pivot, src_lang="en",
                batch_size=batch_size, max_length=max_length,
                do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature
            )
            new_rows.extend([{text_col: t, label_col: label} for t in bt_texts])

    df_bal = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # Expand
    if factor and factor > 0.0:
        if verbose:
            print(f"\nExpanding each class by factor={factor} ...")
        extra = []
        for label, subset in df_bal.groupby(label_col, sort=False):
            n_new = int(len(subset) * factor)
            if n_new <= 0:
                continue
            to_bt = subset.sample(n=n_new, replace=True, random_state=random_state)[text_col].tolist()
            pivot = rng.choice(target_langs)
            if verbose:
                print(f"Class '{label}': +{n_new} via {pivot.upper()}")
            bt_texts = back_translate_batch(
                to_bt, mid_lang=pivot, src_lang="en",
                batch_size=batch_size, max_length=max_length,
                do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature
            )
            extra.extend([{text_col: t, label_col: label} for t in bt_texts])
        if extra:
            df_bal = pd.concat([df_bal, pd.DataFrame(extra)], ignore_index=True)

    return df_bal.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

# ----------------- Train-Only Sanitization Helpers ----------------
def filter_ultra_near_dups_vs_originals(
    originals: pd.DataFrame,
    augmented: pd.DataFrame,
    text_col: str = TEXT_COL,
    sim_thresh: float = 0.98,
) -> pd.DataFrame:
    """Keep augmented rows whose max cosine-sim to any original < sim_thresh."""
    if augmented.empty:
        return augmented.reset_index(drop=True)

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2)
    X_orig = vec.fit_transform(originals[text_col].astype(str).tolist())
    X_aug  = vec.transform(augmented[text_col].astype(str).tolist())

    sims = cosine_similarity(X_aug, X_orig, dense_output=False)
    keep = []
    for i in range(X_aug.shape[0]):
        row = sims.getrow(i)
        max_sim = row.data.max() if row.data.size else 0.0
        keep.append(max_sim < sim_thresh)
    return augmented.loc[keep].reset_index(drop=True)

def sanitize_augmented_training(
    train_original: pd.DataFrame,
    train_aug_full: pd.DataFrame,
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
    near_dup_thresh: float = 0.98,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Keep ALL originals; drop exact dups and ultra-near paraphrases from augmented rows.
    """
    orig = train_original[[text_col, label_col]].copy()
    aug  = train_aug_full[[text_col, label_col]].copy()

    # Remove exact duplicates of originals from augmented set
    orig_norm = set(orig[text_col].map(normalize_text))
    aug["_norm"] = aug[text_col].map(normalize_text)
    aug_only = aug[~aug["_norm"].isin(orig_norm)].drop(columns=["_norm"]).reset_index(drop=True)

    # Remove ultra-near paraphrases vs originals; de-dup among augs; merge
    aug_only = filter_ultra_near_dups_vs_originals(orig, aug_only, text_col=text_col, sim_thresh=near_dup_thresh)
    aug_only = drop_exact_dups(aug_only, text_col=text_col)

    out = pd.concat([orig, aug_only], ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

# --------------------- MLM Augmentation (RoBERTa) ----------------
@torch.inference_mode()
def mlm_augment_roberta(
    df_train: pd.DataFrame,
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
    factor: float = 0.5,
    mask_prob: float = 0.15,
    max_length: int = 256,
    top_k: int = 5,
    batch_size: int = 16,
    random_state: int = 42,
    model_name: str = "roberta-base",
) -> pd.DataFrame:
    """Create ~factor * len(df_train) augmented rows via MLM; preserve class ratios."""
    rng = np.random.RandomState(random_state)
    tok = AutoTokenizer.from_pretrained(model_name)
    mlm = AutoModelForMaskedLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlm.to(device).eval()

    new_rows = []
    for label, subset in df_train.groupby(label_col, sort=False):
        n_new = int(len(subset) * factor)
        if n_new <= 0:
            continue

        sampled = subset.sample(n=n_new, replace=True, random_state=random_state)
        texts = sampled[text_col].tolist()
        labs  = sampled[label_col].tolist()

        for i in tqdm(range(0, n_new, batch_size), desc=f"MLM augment [{label}]"):
            batch_txts = texts[i:i+batch_size]
            enc = tok(batch_txts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

            input_ids = enc["input_ids"].clone()
            attn = enc["attention_mask"]
            special = torch.isin(input_ids, torch.tensor(tok.all_special_ids, device=device))
            cand = (attn == 1) & (~special)

            # Build a per-sequence mask (~mask_prob)
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for r in range(input_ids.size(0)):
                idx = torch.where(cand[r])[0].tolist()
                if not idx:
                    continue
                k = max(1, int(len(idx) * mask_prob))
                chosen = rng.choice(idx, size=k, replace=False)
                mask[r, chosen] = True

            masked_ids = input_ids.clone()
            masked_ids[mask] = tok.mask_token_id

            logits = mlm(masked_ids, attention_mask=attn).logits
            probs  = torch.softmax(logits, dim=-1)
            topk_prob, topk_idx = torch.topk(probs, k=top_k, dim=-1)

            # Sample replacements from top-k
            for r in range(masked_ids.size(0)):
                pos = torch.where(mask[r])[0]
                for t in pos:
                    cand_ids = topk_idx[r, t]
                    cand_p   = topk_prob[r, t] / topk_prob[r, t].sum()
                    choice = torch.multinomial(cand_p, 1)
                    masked_ids[r, t] = cand_ids[choice]

            aug_texts = tok.batch_decode(masked_ids, skip_special_tokens=True)
            for t_aug, y in zip(aug_texts, labs[i:i+batch_size]):
                new_rows.append({text_col: t_aug, label_col: y})

    aug_df = pd.DataFrame(new_rows)
    return drop_exact_dups(aug_df, text_col=text_col)

def mlm_augment_and_clean(
    train_original: pd.DataFrame,
    train_current: pd.DataFrame,
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
    factor: float = 0.5,
    mask_prob: float = 0.15,
    max_length: int = 256,
    top_k: int = 5,
    batch_size: int = 16,
    random_state: int = 42,
    model_name: str = "roberta-base",
    near_dup_thresh: float = 0.98
) -> pd.DataFrame:
    """Augment with MLM, then sanitize vs ORIGINAL train, merge with current."""
    mlm_aug = mlm_augment_roberta(
        df_train=train_current,
        text_col=text_col, label_col=label_col,
        factor=factor, mask_prob=mask_prob, max_length=max_length,
        top_k=top_k, batch_size=batch_size, random_state=random_state, model_name=model_name
    )
    mlm_aug = drop_exact_dups(mlm_aug, text_col=text_col)
    mlm_aug = filter_ultra_near_dups_vs_originals(train_original[[text_col, label_col]], mlm_aug, text_col=text_col, sim_thresh=near_dup_thresh)

    out = pd.concat([train_current[[text_col, label_col]], mlm_aug], ignore_index=True)
    out = drop_exact_dups(out, text_col=text_col)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

# --------------------- EDA Augmentation ------------------------
def _get_synonyms(word: str) -> List[str]:
    syns = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            w = lemma.name().replace('_', ' ')
            if w.lower() != word.lower():
                syns.add(w)
    return [w for w in syns if re.search(r"[A-Za-z]", w)]

def _synonym_replacement(words: List[str], n: int, rng: np.random.RandomState) -> List[str]:
    new_words = words.copy()
    candidates = [w for w in set(words) if re.match(r"[A-Za-z]", w)]
    rng.shuffle(candidates)
    replaced = 0
    for w in candidates:
        syns = _get_synonyms(w)
        if not syns:
            continue
        synonym = rng.choice(syns)
        new_words = [synonym if x == w else x for x in new_words]
        replaced += 1
        if replaced >= n:
            break
    return new_words

def _random_insertion(words: List[str], n: int, rng: np.random.RandomState) -> List[str]:
    new_words = words.copy()
    for _ in range(n):
        candidates = [w for w in new_words if re.match(r"[A-Za-z]", w)]
        if not candidates:
            break
        w = rng.choice(candidates)
        syns = _get_synonyms(w)
        if not syns:
            continue
        synonym = rng.choice(syns)
        pos = rng.randint(0, len(new_words) + 1)
        new_words.insert(pos, synonym)
    return new_words

def _random_swap(words: List[str], n: int, rng: np.random.RandomState) -> List[str]:
    new_words = words.copy()
    for _ in range(n):
        if len(new_words) < 2:
            break
        i, j = rng.randint(0, len(new_words)), rng.randint(0, len(new_words))
        new_words[i], new_words[j] = new_words[j], new_words[i]
    return new_words

def _random_deletion(words: List[str], p: float, rng: np.random.RandomState) -> List[str]:
    if len(words) == 1:
        return words
    kept = [w for w in words if rng.rand() > p]
    return kept if kept else [rng.choice(words)]

def eda_augment_sentence(
    sentence: str,
    alpha_sr: float, alpha_ri: float, alpha_rs: float, p_rd: float,
    ops_per_sample: int,
    rng: np.random.RandomState
) -> str:
    """Apply 1–ops_per_sample EDA ops; each op size scales with sentence length."""
    words = sentence.split()
    if not words:
        return sentence
    n = len(words)
    n_sr = max(1, int(alpha_sr * n)) if alpha_sr > 0 else 0
    n_ri = max(1, int(alpha_ri * n)) if alpha_ri > 0 else 0
    n_rs = max(1, int(alpha_rs * n)) if alpha_rs > 0 else 0

    ops = []
    if n_sr > 0: ops.append(("sr", n_sr))
    if n_ri > 0: ops.append(("ri", n_ri))
    if n_rs > 0: ops.append(("rs", n_rs))
    if p_rd  > 0: ops.append(("rd", p_rd))
    if not ops:
        return sentence

    chosen = np.random.RandomState(rng.randint(0, 10**9)).choice(len(ops), size=min(ops_per_sample, len(ops)), replace=False)
    new_words = words
    for idx in chosen:
        op, val = ops[idx]
        if op == "sr":
            new_words = _synonym_replacement(new_words, val, rng)
        elif op == "ri":
            new_words = _random_insertion(new_words, val, rng)
        elif op == "rs":
            new_words = _random_swap(new_words, val, rng)
        elif op == "rd":
            new_words = _random_deletion(new_words, val, rng)
    return " ".join(new_words)

def eda_augment(
    df_train: pd.DataFrame,
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
    factor: float = 0.5,
    alpha_sr: float = 0.05,
    alpha_ri: float = 0.05,
    alpha_rs: float = 0.05,
    p_rd: float = 0.05,
    ops_per_sample: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Produce ~factor * len(df_train) augmented rows, preserving class ratios."""
    rng = np.random.RandomState(random_state)
    rows = []
    for label, subset in df_train.groupby(label_col, sort=False):
        n_new = int(len(subset) * factor)
        if n_new <= 0:
            continue
        sampled = subset.sample(n=n_new, replace=True, random_state=random_state)
        for s in tqdm(sampled[text_col].tolist(), desc=f"EDA augment [{label}]"):
            aug = eda_augment_sentence(s, alpha_sr, alpha_ri, alpha_rs, p_rd, ops_per_sample, rng)
            rows.append({text_col: aug, label_col: label})
    return drop_exact_dups(pd.DataFrame(rows), text_col=text_col)

def eda_augment_and_clean(
    train_original: pd.DataFrame,
    train_current: pd.DataFrame,
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
    factor: float = 0.5,
    alpha_sr: float = 0.05,
    alpha_ri: float = 0.05,
    alpha_rs: float = 0.05,
    p_rd: float = 0.05,
    ops_per_sample: int = 2,
    random_state: int = 42,
    near_dup_thresh: float = 0.98
) -> pd.DataFrame:
    """EDA augment from `train_current`, clean vs ORIGINAL train, merge, final de-dup."""
    eda_aug = eda_augment(
        df_train=train_current, text_col=text_col, label_col=label_col,
        factor=factor, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs,
        p_rd=p_rd, ops_per_sample=ops_per_sample, random_state=random_state
    )
    eda_aug = drop_exact_dups(eda_aug, text_col=text_col)
    eda_aug = filter_ultra_near_dups_vs_originals(train_original[[text_col, label_col]], eda_aug, text_col=text_col, sim_thresh=near_dup_thresh)

    out = pd.concat([train_current[[text_col, label_col]], eda_aug], ignore_index=True)
    out = drop_exact_dups(out, text_col=text_col)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


# Full pipeline
def run_full_augmentation_pipeline(
    df_raw: pd.DataFrame,
    text_col: str = "statement",
    label_col: str = "verdict",
    # ---- global cleaning & split ----
    sim_threshold_global: float = 0.95,
    train_frac: float = 0.7, val_frac: float = 0.15, test_frac: float = 0.15,
    random_state: int = 42,
    # ---- back-translation (balancing & expansion) ----
    bt_balance_langs = "fr",
    bt_balance_factor: float = 0.0,         # 0 => balance only (no expansion)
    bt_balance_do_sample: bool = False,
    bt_balance_batch_size: int = 32,
    bt_expand_langs = "es",
    bt_expand_factor: float = 1.0,          # ≈ +100% per-class expansion after balance
    bt_expand_do_sample: bool = True,
    bt_expand_top_k: int = 50,
    bt_expand_top_p: float = 0.95,
    bt_expand_temperature: float = 1.0,
    bt_expand_batch_size: int = 16,
    bt_clean_near_dup_thresh: float = 0.98,
    # ---- MLM augmentation (RoBERTa) ----
    mlm_enable: bool = True,
    mlm_factor: float = 0.5,                # +50% per class
    mlm_mask_prob: float = 0.15,
    mlm_top_k: int = 5,
    mlm_max_len: int = 256,
    mlm_batch_size: int = 16,
    mlm_model_name: str = "roberta-base",
    # ---- EDA augmentation ----
    eda_enable: bool = True,
    eda_factor: float = 0.5,                # +50% per class
    eda_alpha_sr: float = 0.05,
    eda_alpha_ri: float = 0.05,
    eda_alpha_rs: float = 0.05,
    eda_p_rd: float = 0.05,
    eda_ops_per_sample: int = 2,
    # ---- sanitation thresholds ----
    near_dup_thresh_post_mlm: float = 0.98,
    near_dup_thresh_post_eda: float = 0.98,
    # ---- verbosity ----
    verbose_bt: bool = True
):
    """
    Run: global clean -> stratified split -> back-translation (balance + expand) -> sanitize
         -> (optional) MLM + sanitize -> (optional) EDA + sanitize.

    Returns
    -------
    train_final : pd.DataFrame    # final training set after all augmentations + cleaning
    val_df      : pd.DataFrame
    test_df     : pd.DataFrame
    intermediates : dict          # useful checkpoints if you want to inspect
    """
    # 0) Global clean (exact + near dups) and split
    df_clean = clean_statement_duplicates(df_raw, text_col=text_col, sim_threshold=sim_threshold_global)
    train_df, val_df, test_df = stratified_split(
        df_clean, label_col=label_col,
        train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
        random_state=random_state
    )

    # 1) Back-translation: balance (factor=0.0) using pivot(s), then expansion
    train_bal = backtranslate_balance_expand_marian(
        train_df,
        text_col=text_col, label_col=label_col,
        target_langs=bt_balance_langs,
        factor=bt_balance_factor,
        random_state=random_state,
        batch_size=bt_balance_batch_size,
        do_sample=bt_balance_do_sample,
        verbose=verbose_bt
    )

    train_bal_expanded = backtranslate_balance_expand_marian(
        train_bal,  # NOTE: expansion uses the original class distribution reference
        text_col=text_col, label_col=label_col,
        target_langs=bt_expand_langs,
        factor=bt_expand_factor,
        random_state=random_state,
        batch_size=bt_expand_batch_size,
        do_sample=bt_expand_do_sample,
        top_k=bt_expand_top_k, top_p=bt_expand_top_p, temperature=bt_expand_temperature,
        verbose=verbose_bt
    )

    # 2) Sanitize BT output against original train (keep all originals)
    train_bt_clean = sanitize_augmented_training(
        train_original=train_df,
        train_aug_full=train_bal_expanded,
        text_col=text_col, label_col=label_col,
        near_dup_thresh=bt_clean_near_dup_thresh,
        random_state=random_state
    )

    # 3) MLM augmentation (+ sanitize) on the BT-cleaned train
    if mlm_enable:
        train_mlm_clean = mlm_augment_and_clean(
            train_original=train_df,
            train_current=train_bt_clean,
            text_col=text_col, label_col=label_col,
            factor=mlm_factor, mask_prob=mlm_mask_prob,
            max_length=mlm_max_len, top_k=mlm_top_k,
            batch_size=mlm_batch_size, random_state=random_state,
            model_name=mlm_model_name,
            near_dup_thresh=near_dup_thresh_post_mlm
        )
    else:
        train_mlm_clean = train_bt_clean

    # 4) EDA augmentation (+ sanitize) on the MLM-cleaned train
    if eda_enable:
        train_final = eda_augment_and_clean(
            train_original=train_df,
            train_current=train_mlm_clean,
            text_col=text_col, label_col=label_col,
            factor=eda_factor,
            alpha_sr=eda_alpha_sr, alpha_ri=eda_alpha_ri, alpha_rs=eda_alpha_rs, p_rd=eda_p_rd,
            ops_per_sample=eda_ops_per_sample,
            random_state=random_state,
            near_dup_thresh=near_dup_thresh_post_eda
        )
    else:
        train_final = train_mlm_clean

    intermediates = {
        "df_clean": df_clean,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "train_bal": train_bal,
        "train_bal_expanded": train_bal_expanded,
        "train_bt_clean": train_bt_clean,
        "train_mlm_clean": train_mlm_clean,
    }
    return train_final, val_df, test_df, intermediates
