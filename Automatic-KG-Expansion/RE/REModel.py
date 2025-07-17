# REModel.py - Relation Extraction for DeepKE-2.2.7 Transformer
from pathlib import Path
import pickle, csv, argparse
from types import SimpleNamespace
import hydra.utils, torch
from deepke.relation_extraction.standard import models
from deepke.relation_extraction.standard.tools import (
    Serializer, _serialize_sentence, _convert_tokens_into_index,
    _add_pos_seq, _handle_relation_data
)
from deepke.relation_extraction.standard.utils import load_pkl

# Configuration paths
BASE_DIR      = Path(__file__).resolve().parent
ROOT_DIR      = BASE_DIR.parent
CKPT_DIR      = BASE_DIR / "checkpoints"
DATA_DIR      = BASE_DIR / "data"
INPUT_FILE    = ROOT_DIR / "re_input.txt"
DEFAULT_SAMPLE= ("门诊部应设在靠近医院入口处",
                 "门诊部",  "SPC",
                 "医院交通入口处", "SITE")
POS_LIMIT     = 50

# Set DeepKE working directory
hydra.utils.get_original_cwd = lambda: str(BASE_DIR)

# Auto-detect checkpoint file
ckpt = next((p for p in CKPT_DIR.glob("*.pth")), None)
if not ckpt:
    raise FileNotFoundError(f"未在 {CKPT_DIR} 找到 *.pth")
CKPT_NAME = ckpt.name

# Load relation mappings
rel_path = DATA_DIR / "relation.csv"
with open(rel_path, newline='', encoding="utf-8") as f:
    rels = _handle_relation_data(list(csv.DictReader(f)))
rel_names = list(rels.keys())

# Load vocabulary
vocab = load_pkl(DATA_DIR / "vocab.pkl", verbose=False)

# Initialize serializer
serializer = Serializer(do_chinese_split=True)

def build_cfg():
     """Build configuration for model with position embeddings"""
    word_dim  = 60
    pos_limit = 30
    pos_dim   = 10
    hidden_sz = word_dim + 2 * pos_dim
    return SimpleNamespace(
        # Basic paths
        model_name      = "transformer",
        output_dir      = str(CKPT_DIR),
        model_save_name = CKPT_NAME,

        # Vocabulary and position vectors
        vocab_size      = vocab.count,
        pos_limit       = pos_limit,
        pos_size        = 2 * pos_limit + 2,

        # Preprocessing
        chinese_split   = True,
        replace_entity_with_type  = True,
        replace_entity_with_scope = False,
        use_gpu         = False,

        # Embedding parameters
        word_dim        = word_dim,
        pos_dim         = pos_dim,
        dim_strategy    = "sum",

        # Transformer parameters
        num_heads           = 6,
        num_hidden_layers   = 4,
        intermediate_size   = 256,
        dropout             = 0.2,
        layer_norm_eps      = 1e-12,
        hidden_act          = "gelu",
        output_attentions   = True,
        output_hidden_states= True,

        # Additional required fields
        hidden_size     = hidden_sz,
        num_relations   = len(rel_names)
    )

def preprocess(sentence, h, t, h_type, t_type, cfg):
    """Preprocess single data sample for DeepKE input"""
    data = [{
        "sentence": sentence, "head": h, "tail": t,
        "head_type": h_type, "tail_type": t_type
    }]
    _serialize_sentence(data, serializer.serialize, cfg)
    _convert_tokens_into_index(data, vocab)
    _add_pos_seq(data, cfg)
    ex = data[0]
    # Padding to 512 tokens
    max_len = 512
    pad = lambda x: x + [0] * (max_len - len(x))
    sample = {
        "word": torch.tensor([pad(ex["token2idx"][:max_len])]),
        "lens": torch.tensor([min(ex["seq_len"], max_len)]),
        "head_pos": torch.tensor([pad(ex["head_pos"][:max_len])]),
        "tail_pos": torch.tensor([pad(ex["tail_pos"][:max_len])])
    }
    return sample

def load_model(cfg):
    model = models.Transformer(cfg)
    model.load_state_dict(
        torch.load(CKPT_DIR / ckpt.name, map_location="cpu")
    )
    model.eval()
    return model

def predict_relation(model, sample):
    """Predict relation for given sample"""
    with torch.no_grad():
        logits = model({k: v for k, v in sample.items()})
        prob   = torch.softmax(logits, dim=-1)[0]
        idx    = prob.argmax().item()
        return rel_names[idx], prob[idx].item()

def parse_batch_file():
    """Parse batch input file with format: sentence<TAB>head<TAB>head_type<TAB>tail<TAB>tail_type"""
    if INPUT_FILE.exists():
        with open(INPUT_FILE, encoding="utf-8") as f:
            for ln in f:
                parts = [p.strip() for p in ln.rstrip("\n").split("\t")]
                if len(parts) == 5:
                    yield tuple(parts)
    else:
        yield DEFAULT_SAMPLE

if __name__ == "__main__":
    cfg   = build_cfg()
    model = load_model(cfg)

    for i, (sent, h, h_t, t, t_t) in enumerate(parse_batch_file(), 1):
        sample = preprocess(sent, h, t, h_t, t_t, cfg)
        rel, conf = predict_relation(model, sample)
        print(f"\n—— 样本 {i} ———————————————")
        print("句子 :", sent)
        print(f"头实体: {h} ({h_t})")
        print(f"尾实体: {t} ({t_t})")
        print(f"预测关系: {rel}")