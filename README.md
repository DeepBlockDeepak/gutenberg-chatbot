# CSC525: NLP Chatbot Portfolio Project

This project showcases a locally trained chatbot that uses a character-level LSTM (PyTorch) trained on locally stored text files from a public-domain work (*Pride and Prejudice* is the training corpus for this pre-trained and pre-loaded model). The application runs on **Taipy** for the front-end web application and chat interface.

## Requirements

- `Python 3.10` or higher
- `uv` (for environment management and running tasks)

All necessary Python packages will be installed automatically with **uv**.

## Installation and Running

After installing `uv` on your system...

From project root, you can set up dependencies + the environment, and start the Chatbot application all with:
```sh
uv run main.py
```

(combines the use of `uv sync` as a precursor to the driver)

The local address which appears in terminal is `http://127.0.0.1:5001`, and will redirect you to the chatbot browser application session.


### Interface
![title](static/RNN%20Chat%20Demo%20screenshot.png)


## Optional Training Functionality
If you'd like to, you can engage the training script with:
```bash
uv run src/gutenberg_chatbot/training/training.py
```

This script will read the specified .txt file in `datasets/`, train the `RNNModel`, and save the resulting checkpoint to the `models/` directory.
By default, `training.py` uses the following variable to store the training corpus text file:
```py
training_corpus_text_path="datasets/pride_and_prejudice.txt"
```

If you'd like, you can save your own file in a similar manner to train and interact with a different style of chat model.

## Reference:
- Initial Taipy Exploration and front end inspiration was taken from:
[Taipy's Creating an LLM ChatBot Documentation Page](https://docs.taipy.io/en/release-3.0/knowledge_base/tutorials/chatbot/)
- [Project Gutenberg](https://www.gutenberg.org/) for public-domain training corpus.