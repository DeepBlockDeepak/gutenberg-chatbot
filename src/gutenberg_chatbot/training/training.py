import json
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gutenberg_chatbot.model import (
    RNNModel,
    TextDataset,
    load_model,
    save_model,
    device,
)


def save_vocab_mappings(c2ix, ix2c, path):
    with open(path, "w") as f:
        json.dump({"c2ix": c2ix, "ix2c": ix2c}, f)


def load_vocab_mappings(path):
    with open(path, "r") as f:
        data = json.load(f)

    c2ix = data["c2ix"]  # {"i": 67, "m": 68, ...},
    ix2c_str = data["ix2c"]  # {"67": "i", "68": "m", ...}

    # c2ix = {k: v for k, v in c2ix_str.items()}

    # convert string keys to integer keys:
    ix2c = {int(k): v for k, v in ix2c_str.items()}

    return c2ix, ix2c


def train_rnn(
    text_path: str,
    checkpoint_path: str,
    seq_length: int = 24,
    batch_size: int = 48,
    embedding_dim: int = 32,
    hidden_dim: int = 128,
    num_layers: int = 1,
    learning_rate: float = 0.003,
    epochs: int = 10,
    patience: int = 3,
):
    """
    Train or resume training of the RNN model on a text dataset.

    Args:
        text_path (str): Path to the raw text file.
        checkpoint_path (str): Where to save/load the model checkpoint.
        seq_length, batch_size, etc: Hyperparameters.
        epochs: Max number of epochs to train.
        patience: Early-stopping patience.

    Returns:
        (model, c2ix, ix2c): The trained model and dictionaries for token mapping.
    """
    # load text
    with open(text_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # build character dictionaries
    tokenized_text = list(raw_text)
    unique_chars = sorted(list(set(tokenized_text)))
    c2ix = {ch: i for i, ch in enumerate(unique_chars)}
    ix2c = {i: ch for ch, i in c2ix.items()}
    vocab_size = len(unique_chars)

    # save the character/token mapping
    # derive corpus_name from text_path
    corpus_name = os.path.splitext(os.path.basename(text_path))[0]
    save_vocab_mappings(c2ix, ix2c, f"models/{corpus_name}_vocab.json")

    token_ids = [c2ix[ch] for ch in tokenized_text]

    # create dataset and dataloader
    dataset = TextDataset(token_ids, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # instantiate or load model
    model = RNNModel(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        res = load_model(checkpoint_path, RNNModel, device, optimizer=optimizer)
        if len(res) == 3:
            model, start_epoch, optimizer = res
        else:
            model, start_epoch = res
        print(f"Resuming training from epoch {start_epoch+1}")

    criterion = torch.nn.CrossEntropyLoss()

    # training loop with early stopping
    best_loss = np.inf
    epochs_no_improve = 0

    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(X)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0

            # After training or in the best checkpoint condition:
            save_model(
                model,
                optimizer,
                epoch + 1,
                checkpoint_path,
                (vocab_size, embedding_dim, hidden_dim, num_layers),
                corpus_name,
            )
            print(f"New best model saved (loss={best_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"Stopping early at epoch {epoch+1}. Best loss: {best_loss:.4f}")
            break

    return model, c2ix, ix2c


if __name__ == "__main__":
    model, c2ix, ix2c = train_rnn(
        text_path="datasets/pride_and_prejudice.txt",
        checkpoint_path="models/rnn_model.pth",
        epochs=10,
    )
