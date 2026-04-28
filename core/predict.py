# predict.py — load .pt checkpoint, run NER, extract entity spans

import torch
from transformers import BertTokenizerFast
from core.model import NERModel, id2label

MODEL_PATH = "tempeval3_ner_final.pt"
MAX_LEN    = 128

_model     = None
_tokenizer = None
_device    = None


def load_model(model_path: str = MODEL_PATH):
    global _model, _tokenizer, _device
    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
    _model     = NERModel().to(_device)
    state      = checkpoint.get("model_state_dict", checkpoint)
    _model.load_state_dict(state)
    _model.eval()
    ep  = checkpoint.get("epoch", "?")
    f1  = checkpoint.get("val_f1", 0)
    print(f"[NER] Loaded — epoch {ep}  val_F1={f1:.4f}  device={_device}")
    return _model


def predict_sentence(text: str):
    """Returns list of (token, label) for every word-token in text."""
    global _model, _tokenizer, _device
    if _model is None:
        load_model()
    tokens = text.split()
    if not tokens:
        return []
    enc = _tokenizer(
        tokens, is_split_into_words=True,
        return_tensors="pt", truncation=True,
        max_length=MAX_LEN, padding=False,
    ).to(_device)
    with torch.no_grad():
        preds = _model(enc["input_ids"], enc["attention_mask"])[0]
    word_ids = enc.word_ids(batch_index=0)
    result, seen = [], set()
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in seen:
            continue
        seen.add(word_idx)
        label = id2label[preds[idx]] if idx < len(preds) else "O"
        result.append((tokens[word_idx], label))
    return result


def extract_entities(token_labels):
    """Groups consecutive B/I tokens into entity dicts."""
    entities, current, etype = [], [], None
    for token, label in token_labels:
        if label.startswith("B-"):
            if current:
                entities.append({"text": " ".join(current), "type": etype})
            current, etype = [token], label[2:]
        elif label.startswith("I-") and current:
            current.append(token)
        else:
            if current:
                entities.append({"text": " ".join(current), "type": etype})
            current, etype = [], None
    if current:
        entities.append({"text": " ".join(current), "type": etype})
    return entities
