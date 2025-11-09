import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# -------- Utilities --------
def ensure_text_full(df: pd.DataFrame, text_priority=None) -> pd.DataFrame:
    if 'text_full' in df.columns:
        return df
    if text_priority is None:
        text_priority = ['text', 'text_full', 'content', 'article', 'body', 'title']
    for cand in text_priority:
        if cand in df.columns:
            out = df.copy()
            out['text_full'] = out[cand].astype(str)
            return out
    obj_cols = [c for c in df.columns if df[c].dtype == 'O']
    if obj_cols:
        out = df.copy()
        out['text_full'] = out[obj_cols[0]].astype(str)
        return out
    raise ValueError("No suitable text column to create 'text_full'.")

def remove_exact_duplicates(df: pd.DataFrame, text_col: str = 'text_full') -> pd.DataFrame:
    return df.drop_duplicates(subset=[text_col]).reset_index(drop=True)

def remove_near_duplicates(
    df: pd.DataFrame,
    text_col: str = 'text_full',
    tfidf_radius: float = 0.05,
    min_df: int = 2,
    max_df: float = 0.9,
    ngram_range: Tuple[int, int] = (1, 2),
    stop_words: str = 'english'
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    texts = df[text_col].astype(str).tolist()
    vectorizer = TfidfVectorizer(
        stop_words=stop_words, min_df=min_df, max_df=max_df, ngram_range=ngram_range
    )
    X = vectorizer.fit_transform(texts)

    nn = NearestNeighbors(radius=tfidf_radius, metric='cosine', n_jobs=-1)
    nn.fit(X)

    keep = np.ones(len(df), dtype=bool)
    nbrs = nn.radius_neighbors(X, return_distance=False)

    seen = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        if seen[i]:
            keep[i] = False
            continue
        seen[nbrs[i]] = True
        keep[i] = True

    return df.loc[keep].reset_index(drop=True)

def _dedup_across_splits_exact(train_df, val_df, test_df, text_col='text_full'):
    # Normalize
    for df in (train_df, val_df, test_df):
        df[text_col] = df[text_col].astype(str).str.strip()

    train_texts = set(train_df[text_col])
    val_df = val_df.loc[~val_df[text_col].isin(train_texts)].reset_index(drop=True)

    val_texts = set(val_df[text_col])
    test_df = test_df.loc[~test_df[text_col].isin(train_texts.union(val_texts))].reset_index(drop=True)
    return train_df, val_df, test_df

def _dedup_across_splits_near(
    train_df, val_df, test_df,
    text_col='text_full',
    tfidf_radius: float = 0.05,
    min_df: int = 2,
    max_df: float = 0.9,
    ngram_range: Tuple[int, int] = (1, 2),
    stop_words: str = 'english'
):
    """
    Remove near-duplicates ACROSS splits with priority: train > val > test.
    Greedy keep-first according to (priority, original_order).
    """
    # Tag splits and stack
    def _prep(df, split):
        df = df.copy()
        df[text_col] = df[text_col].astype(str).str.strip()
        df['_split'] = split
        return df

    tr = _prep(train_df, 'train')
    va = _prep(val_df, 'val')
    te = _prep(test_df, 'test')
    all_df = pd.concat([tr, va, te], ignore_index=True)

    # Priority order
    pri_map = {'train': 0, 'val': 1, 'test': 2}
    all_df['_pri'] = all_df['_split'].map(pri_map)

    # TF-IDF
    texts = all_df[text_col].tolist()
    if len(texts) == 0:
        return train_df, val_df, test_df

    vectorizer = TfidfVectorizer(
        stop_words=stop_words, min_df=min_df, max_df=max_df, ngram_range=ngram_range
    )
    X = vectorizer.fit_transform(texts)

    nn = NearestNeighbors(radius=tfidf_radius, metric='cosine', n_jobs=-1)
    nn.fit(X)
    nbrs = nn.radius_neighbors(X, return_distance=False)

    # Process in priority order; drop later neighbors
    order = np.lexsort((np.arange(len(all_df)), all_df['_pri'].values))  # sort by _pri then by original order
    drop = np.zeros(len(all_df), dtype=bool)

    # map index -> position in 'order' (for "later only" dropping)
    pos = np.empty(len(all_df), dtype=int)
    pos[order] = np.arange(len(all_df))

    for idx in order:
        if drop[idx]:
            continue
        # drop only neighbors that come later in priority/order
        for j in nbrs[idx]:
            if pos[j] > pos[idx]:
                drop[j] = True

    kept = all_df.loc[~drop].drop(columns=['_pri'])
    # Split back
    train_out = kept[kept['_split'] == 'train'].drop(columns=['_split']).reset_index(drop=True)
    val_out   = kept[kept['_split'] == 'val'  ].drop(columns=['_split']).reset_index(drop=True)
    test_out  = kept[kept['_split'] == 'test' ].drop(columns=['_split']).reset_index(drop=True)
    return train_out, val_out, test_out

# -------- Main pipeline (single DataFrame) --------
def PreprocessDataOneDF(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    tfidf_radius: float = 0.05,
    *,
    dedup_within_label: bool = True,
    post_cross_exact_dedup: bool = True,
    post_cross_near_dedup: bool = False,
    post_radius: float = 0.05
):
    """
    Pipeline for a single df with columns: 'text' and 'label' (binary {0,1}).
      - Ensure 'text_full'
      - Remove exact duplicates (per-label or global)
      - Remove near-duplicates (per-label or global)
      - Stratified split into train/val/test
      - Remove exact duplicates across splits (train > val > test)
      - (Optional) Remove near-duplicates across splits with TF-IDF radius

    Returns train_df, val_df, test_df with ['text_full','label'].
    """
    s = train_frac + val_frac + test_frac
    if not np.isclose(s, 1.0, atol=1e-6):
        raise ValueError(f"Fractions must sum to 1.0; got {s}")

    if 'label' not in df.columns or 'text' not in df.columns:
        raise ValueError("Input df must contain 'text' and 'label' columns.")
    if not set(np.unique(df['label'])).issubset({0,1}):
        raise ValueError("'label' must be binary {0,1}.")

    # 1) Ensure text_full
    df = df.copy()
    df['text'] = df['text'].astype(str)
    df = ensure_text_full(df, text_priority=['text', 'text_full'])
    df = df[['text_full', 'label']].copy()
    df['text_full'] = df['text_full'].astype(str).str.strip()
    df['label'] = df['label'].astype(int)

    # 2) Exact duplicates
    if dedup_within_label:
        df_pos = remove_exact_duplicates(df[df['label'] == 1], 'text_full')
        df_neg = remove_exact_duplicates(df[df['label'] == 0], 'text_full')
    else:
        df = remove_exact_duplicates(df, 'text_full')
        df_pos = df[df['label'] == 1].reset_index(drop=True)
        df_neg = df[df['label'] == 0].reset_index(drop=True)

    # 3) Near-duplicates
    if dedup_within_label:
        df_pos = remove_near_duplicates(df_pos, 'text_full', tfidf_radius=tfidf_radius)
        df_neg = remove_near_duplicates(df_neg, 'text_full', tfidf_radius=tfidf_radius)
        df_all = pd.concat([df_pos, df_neg], ignore_index=True)
    else:
        df_all = pd.concat([df_pos, df_neg], ignore_index=True)
        df_all = remove_near_duplicates(df_all, 'text_full', tfidf_radius=tfidf_radius)

    # 4) Stratified split
    y_all = df_all['label'].astype(int).to_numpy()
    trainval_df, test_df = train_test_split(
        df_all, test_size=test_frac, random_state=42, stratify=y_all
    )
    val_frac_of_trainval = val_frac / (train_frac + val_frac)
    y_trainval = trainval_df['label'].astype(int).to_numpy()
    train_df, val_df = train_test_split(
        trainval_df, test_size=val_frac_of_trainval, random_state=42, stratify=y_trainval
    )

    # 5) Final tidy columns
    train_df = train_df[['text_full', 'label']].reset_index(drop=True)
    val_df   = val_df[['text_full', 'label']].reset_index(drop=True)
    test_df  = test_df[['text_full', 'label']].reset_index(drop=True)

    # 6) Post-split cross-split dedup (exact)
    if post_cross_exact_dedup:
        train_df, val_df, test_df = _dedup_across_splits_exact(train_df, val_df, test_df, text_col='text_full')

    # 7) Optional: Post-split cross-split near-dup dedup
    if post_cross_near_dedup:
        train_df, val_df, test_df = _dedup_across_splits_near(
            train_df, val_df, test_df,
            text_col='text_full',
            tfidf_radius=post_radius
        )

    return train_df, val_df, test_df
