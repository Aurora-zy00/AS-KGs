# NERModel.py - Named Entity Recognition for DeepKE-2.2.7
from pathlib import Path
import pickle
from types import SimpleNamespace
import hydra.utils
from deepke.name_entity_re.standard import InferNer

# Configuration paths
BASE_DIR      = Path(__file__).resolve().parent
ROOT_DIR      = BASE_DIR.parent
MODEL_DIR     = BASE_DIR / "checkpoints"
VOCAB_FILE    = BASE_DIR / "vocab.pkl"
SENTENCE_FILE = ROOT_DIR / "test.txt"
DEFAULT_TEXT  = "门诊部应设在靠近医院入口处"


# Set DeepKE working directory
hydra.utils.get_original_cwd = lambda: str(BASE_DIR)

# Auto-detect model file
ckpt_files = list(MODEL_DIR.glob("*.pth")) + list(MODEL_DIR.glob("*.ckpt"))
if not ckpt_files:
    raise FileNotFoundError(f"在 {MODEL_DIR} 未找到 *.pth / *.ckpt")
MODEL_FILE = ckpt_files[0].name

def load_ner():
    """Load DeepKE-2.2.7 LSTM-CRF inference model"""
    with open(VOCAB_FILE, "rb") as f:
        word2id  = pickle.load(f)
        label2id = pickle.load(f)
        id2label = pickle.load(f)

    cfg = SimpleNamespace(
        model_name      = "lstmcrf",
        output_dir      = str(MODEL_DIR),
        model_save_name = MODEL_FILE
    )

    return InferNer(
        str(MODEL_DIR),
        cfg,
        len(word2id),
        len(label2id),
        word2id,
        id2label
    )

def read_sentences():
    """Read sentences from test file or use default"""
    if SENTENCE_FILE.exists():
        with open(SENTENCE_FILE, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if lines:
            return lines
    return [DEFAULT_TEXT]

def extract_entities(token_tag_list):

    entities, buf, cur_type = [], [], None
    for ch, tag in token_tag_list:
        if tag == 'O':
            if buf:
                entities.append((''.join(buf), cur_type))
                buf, cur_type = [], None
            continue
        prefix, e_type = tag.split('-', 1)
        if prefix == 'B' or e_type != cur_type:
            if buf:
                entities.append((''.join(buf), cur_type))
            buf, cur_type = [ch], e_type
        else:  # 'I'
            buf.append(ch)
    if buf:
        entities.append((''.join(buf), cur_type))
    return entities


if __name__ == "__main__":
    ner        = load_ner()
    sentences  = read_sentences()

    for idx, sent in enumerate(sentences, 1):
        print(f"\n—— 句子 {idx} ———————————————")
        print("原文:", sent)
        print("NER :", ner.predict(sent))
