import os
from typing import Optional


import torch
import torch.nn as nn
from torch.utils.data import Dataset

torch.manual_seed(1)

# Ensure models directory exists
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
match (torch.cuda.is_available(), torch.backends.mps.is_available()):
    case (True, _):
        device = torch.device("cuda")
    case (False, True):
        device = torch.device("mps")
    case _:
        device = torch.device("cpu")


def load_model(
    checkpoint_path: str,
    model_class,
    device,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    """
    Loads a saved model checkpoint and optionally restores the optimizer.

    Args:
        checkpoint_path: path to the saved checkpoint.
        model_class: model to load.
        device: device to load the model onto.
        optimizer: optimizer.

    Returns:
        nn.Module: The loaded model.
        int: The last trained epoch.
        torch.optim.Optimizer (optional): The restored optimizer (if provided).
    """
    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] No model checkpoint found at {checkpoint_path}.")
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = model_class(*checkpoint["init_args"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint["epoch"]
    # retrieve corpus_name
    corpus_name = checkpoint.get("corpus_name", "UNKNOWN_CORPUS")

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, epoch, optimizer, corpus_name

    return model, epoch, corpus_name


def save_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_path: str,
    init_args: tuple,
    corpus_name: str,
):
    """
    Saves the model state, optimizer state, and initialization arguments.

    Args:
        model: model to save.
        optimizer: the optimizer used during training.
        epoch: current epoch number.
        checkpoint_path: path to save the checkpoint.
        init_args: arguments used to initialize the model.
        corpus_name: A string identifying which text/corpus was used.
    """
    torch.save(
        {
            "epoch": epoch,
            "init_args": init_args,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "corpus_name": corpus_name,
        },
        checkpoint_path,
    )
    print(f"Model saved to {checkpoint_path}")


class TextDataset(Dataset):
    """Simple PyTorch Dataset for character-based language modeling."""

    def __init__(self, tokenized_text: list[int], seq_length: int):
        """
        Args:
            tokenized_text: The entire text as a list of token IDs.
            seq_length: Number of tokens in the input sequence.
        """
        self.tokenized_text = tokenized_text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.tokenized_text) - self.seq_length

    def __getitem__(self, idx):
        features = torch.tensor(self.tokenized_text[idx : idx + self.seq_length])
        labels = torch.tensor(self.tokenized_text[idx + 1 : idx + self.seq_length + 1])
        return features, labels


class RNNModel(nn.Module):
    """A simple LSTM-based language model."""

    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1
    ):
        """
        Args:
            vocab_size (int): Number of unique tokens.
            embedding_dim (int): Dimension of the embedding vector.
            hidden_dim (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_length).
            hidden (torch.Tensor, optional): Hidden state (if any).

        Returns:
            logits (torch.Tensor): Output logits for each token (batch, seq_length, vocab_size).
            hidden (torch.Tensor): The new hidden state.
        """
        embedded = self.embedding(x)  # Shape: (batch, seq_length, embedding_dim)
        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden


def generate_text(
    model: nn.Module,
    seed_text: str,
    c2ix: dict[str, int],
    ix2c: dict[int, str],
    generation_length: int = 100,
    temperature: float = 1.0,
) -> str:
    """
    Generate text from the model given a seed input.

    Args:
        model: The trained RNNModel.
        seed_text: Seed string for generation.
        c2ix: Dict mapping characters to token IDs.
        ix2c: Dict mapping token IDs to characters.
        generation_length: Number of characters to generate.
        temperature: Sampling temperature; higher values => more randomness.

    Returns:
        str: The generated text.
    """
    model.eval()
    # convert seed text to token ids (unknown chars default to index 0).
    input_ids = [c2ix.get(ch, 0) for ch in seed_text]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    generated_str = ""
    hidden = None

    with torch.no_grad():
        for _ in range(generation_length):
            logits, hidden = model(input_tensor, hidden)
            # only consider the output from the last time step
            last_logits = logits[:, -1, :]
            # adjust logits by the temperature
            last_logits /= temperature
            probs = torch.softmax(last_logits, dim=-1)
            # sample from the distribution
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            next_char = ix2c[next_token_id]
            generated_str += next_char
            # prepare input for next iteration
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)

    return generated_str
