import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

torch.manual_seed(1)

# Ensure models directory exists
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(checkpoint_path: str, model_class, device, optimizer=None):
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

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, epoch, optimizer

    return model, epoch


def save_model(model, optimizer, epoch, checkpoint_path: str, init_args):
    """
    Saves the model state, optimizer state, and initialization arguments.

    Args:
        model: model to save.
        optimizer: the optimizer used during training.
        epoch: current epoch number.
        checkpoint_path: path to save the checkpoint.
        init_args: arguments used to initialize the model.
    """
    torch.save(
        {
            "epoch": epoch,
            "init_args": init_args,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Model saved to {checkpoint_path}")


class TextDataset(Dataset):
    def __init__(self, tokenized_text, seq_length):
        self.tokenized_text = tokenized_text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.tokenized_text) - self.seq_length

    def __getitem__(self, idx):
        features = torch.tensor(self.tokenized_text[idx : idx + self.seq_length])
        labels = torch.tensor(self.tokenized_text[idx + 1 : idx + self.seq_length + 1])
        return features, labels


###############################################################################
# Define the RNN (LSTM) model
###############################################################################
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
        super(RNNModel, self).__init__()
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


###############################################################################
# Training the model
###############################################################################
if __name__ == "__main__":
    print(
        f"Training on {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}"
    )

    # Load data
    with open("datasets/pride_and_prejudice.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenized_text = list(raw_text)
    unique_character_tokens = sorted(list(set(tokenized_text)))
    c2ix = {ch: i for i, ch in enumerate(unique_character_tokens)}
    ix2c = {ix: ch for ch, ix in c2ix.items()}
    vocab_size = len(c2ix)
    tokenized_id_text = [c2ix[c] for c in tokenized_text]

    seq_length = 24
    dataset = TextDataset(tokenized_id_text, seq_length)
    batch_size = 48
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    # Define model and optimizer
    embedding_dim = 32
    hidden_dim = 128
    num_layers = 1
    learning_rate = 0.003
    epochs = 10
    checkpoint_path = os.path.join(MODEL_DIR, "rnn_model.pth")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = RNNModel(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Check if a model exists before training
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}...")
        model, start_epoch, optimizer = load_model(
            checkpoint_path, RNNModel, device, optimizer
        )
        print(f"Resuming training from epoch {start_epoch + 1}")

    # Early stopping
    patience = 3
    best_loss = np.inf
    epochs_no_improve = 0

    # Training loop
    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for batch_features, batch_labels in progress_bar:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(batch_features)
            loss = criterion(logits.view(-1, vocab_size), batch_labels.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")

        # Save model if best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            save_model(
                model,
                optimizer,
                epoch + 1,
                checkpoint_path,
                (vocab_size, embedding_dim, hidden_dim, num_layers),
            )
            print(f"New best model saved with loss: {best_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Stopping early at epoch {epoch + 1}. Best loss: {best_loss:.4f}")
            break  # Exit training loop early


###############################################################################
# Text generation function using the trained model
###############################################################################
def generate_text(
    model: nn.Module,
    start_text: str,
    generation_length: int = 100,
    temperature: float = 1.0,
) -> str:
    """
    Generate text from the model given a seed input.

    Args:
        model (nn.Module): Trained language model.
        start_text (str): Seed text to start generation.
        generation_length (int): Number of characters to generate.
        temperature (float): Sampling temperature; higher values yield more random predictions.

    Returns:
        str: The generated text.
    """
    model.eval()
    # Convert seed text to token ids (if a character is not found, default to index 0)
    input_ids = [c2ix.get(ch, 0) for ch in start_text]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    # generated = start_text
    generated = ""

    with torch.no_grad():
        for _ in range(generation_length):
            logits, hidden = model(input_tensor, hidden)
            # Only consider the output from the last time step
            last_logits = logits[:, -1, :]
            # Adjust logits by the temperature parameter
            last_logits /= temperature
            probabilities = torch.softmax(last_logits, dim=-1)
            # Sample the next character id from the probability distribution
            next_token_id = torch.multinomial(probabilities, num_samples=1).item()
            next_char = ix2c[next_token_id]
            generated += next_char
            # Prepare input tensor for next prediction (shape: [1, 1])
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)

    return generated


###############################################################################
# Prompt the user for input and generate text
###############################################################################
# if __name__ == "__main__":
#     if os.path.exists(checkpoint_path):
#         print(f"Loading trained model for text generation from {checkpoint_path}...")
#         model, _ = load_model(checkpoint_path, RNNModel, device)
#     else:
#         print("No trained model found! Train the model first.")

#     seed_text = input("Enter seed text for generation: ")
#     generated_text = generate_text(
#         model, seed_text, generation_length=200, temperature=0.8
#     )
#     print("\nGenerated text:")
#     print(generated_text)
