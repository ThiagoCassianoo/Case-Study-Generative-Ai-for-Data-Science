"""
data_prep.py
------------
Camada de DOMÍNIO — replica fielmente a lógica de preparação de dados do
notebook original `fine-tune-demo.ipynb` do repositório de referência
(dgodevais/coursera-genai-for-data-science), mas devolve texto puro em vez
do prompt estilo [INST] usado pra um LLM causal — porque aqui vamos treinar
um classificador (encoder), não fazer prompting de um LLM generativo.

Tarefa (idêntica ao original): classificar perguntas do Stack Overflow em
3 classes de qualidade a partir de Título + Corpo + Tags:
  - HQ        : pergunta de alta qualidade, sem edições
  - LQ_EDIT   : baixa qualidade, mas editada pela comunidade (ainda aberta)
  - LQ_CLOSE  : baixa qualidade, fechada pela comunidade
"""

from __future__ import annotations

import re

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["HQ", "LQ_EDIT", "LQ_CLOSE"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def strip_html(text: str) -> str:
    """O campo Body vem em HTML cru (<p>, <code> etc). Remove as tags
    pra não inflar o tamanho do input do tokenizer com ruído estrutural."""
    return re.sub(r"<[^>]+>", " ", str(text)).strip()


def load_and_clean(csv_path: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Sujeira real encontrada no CSV: ~1% das linhas tem Y nulo (não documentado
    # no repo original). Como é o próprio alvo, não dá pra imputar — descartamos.
    n_before = len(df)
    df = df.dropna(subset=["Y"]).copy()
    dropped = n_before - len(df)
    if dropped:
        print(f"[data_prep] Descartadas {dropped} linhas com Y nulo ({dropped/n_before:.1%} do total).")

    if max_rows:
        df = df.iloc[:max_rows].copy()

    df["body_clean"] = df["Body"].apply(strip_html)
    df["tags_clean"] = df["Tags"].str.replace("><", ", ", regex=False).str.strip("<>")

    # Texto único de entrada pro classificador — título pesa mais (repetido),
    # já que é o sinal mais denso de qualidade da pergunta.
    df["text"] = (
        df["Title"] + " [SEP] " + df["Title"] + " [SEP] "
        + df["body_clean"].str.slice(0, 800) + " [SEP] tags: " + df["tags_clean"]
    )
    df["label"] = df["Y"].map(LABEL2ID)
    return df[["Id", "text", "label", "Y"]]


def split_dataframe(df: pd.DataFrame, train_size=0.6, test_size=0.2, random_state=999):
    """Mesma proporção 60/20/20 usada no notebook original (fine-tune-demo.ipynb),
    mantida aqui de propósito pra comparação justa de métricas."""
    train_df, remaining_df = train_test_split(
        df, train_size=train_size, random_state=random_state, stratify=df["label"]
    )
    val_test_ratio = test_size / (1 - train_size)
    val_df, test_df = train_test_split(
        remaining_df, test_size=val_test_ratio, random_state=random_state,
        stratify=remaining_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_and_clean("../train.csv")
    train_df, val_df, test_df = split_dataframe(df)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(df["Y"].value_counts())
    print(train_df.iloc[0]["text"][:300])
