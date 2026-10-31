"""Predykcja modelu distilbert-v1 w Space — tak jak `DistilBertClassifier.predict_proba` w repo.

Tokenizer z artefaktu (`padding`, `truncation`, `max_length` z `metadata.json`, bez
`token_type_ids`), model na CPU w trybie ewaluacji, softmax na logitach. Nazwy klas z `id2label`
w `config.json` artefaktu (ta sama mapa co `configs/label_map.json`, sprawdzana przy zapisie).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "model"))
MAX_TEXT_CHARS = 500  # jak `api.max_text_chars` w repo


@dataclass(frozen=True)
class Prediction:
    intent: str
    intent_id: int
    score: float  # wynik modelu dla klasy (softmax), nie gwarancja trafienia


class IntentModel:
    def __init__(self, directory: Path = MODEL_DIR) -> None:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self.max_len = int(metadata["max_len"])
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True
        ).to("cpu").eval()
        self.id2label = {int(key): name for key, name in self.model.config.id2label.items()}

    def top(self, text: str, k: int = 3) -> list[Prediction]:
        """`k` klas o najwyższym score dla jednej wiadomości, malejąco."""
        inputs = self.tokenizer(
            [text], padding=True, truncation=True, max_length=self.max_len,
            return_token_type_ids=False, return_tensors="pt",
        )
        with torch.inference_mode():
            scores = torch.softmax(self.model(**inputs).logits.float(), dim=-1)[0]
        values, indices = scores.topk(k)
        return [
            Prediction(self.id2label[int(index)], int(index), float(value))
            for value, index in zip(values, indices, strict=True)
        ]
