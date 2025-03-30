import os

from gutenberg_chatbot.model import RNNModel, device, load_model
from gutenberg_chatbot.training.training import load_vocab_mappings


def load_default_model_and_vocab() -> tuple[object, dict, dict]:
    """
    Tries loading the default model checkpoint and vocab.
    Returns (model, c2ix, ix2c). If anything fails, returns (None, {}, {}).
    """
    MODEL_DIR = "models"
    checkpoint_path = os.path.join(MODEL_DIR, "rnn_model.pth")
    result = load_model(checkpoint_path, RNNModel, device)

    # no model found or load failed
    if result is None or result[0] is None:
        print("No model found. Please train first.")
        return None, {}, {}

    model, start_epoch, corpus_name = result
    # load the corresponding vocab to the corpus that was used
    vocab_path = os.path.join(MODEL_DIR, f"{corpus_name}_vocab.json")

    if os.path.exists(vocab_path):
        c2ix, ix2c = load_vocab_mappings(vocab_path)
    else:
        print(
            f"[WARNING] No vocab file {vocab_path} found. The model may fail to generate properly."
        )
        c2ix, ix2c = {}, {}

    return model, c2ix, ix2c
