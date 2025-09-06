## Transformer: Sequence-to-Sequence Neural Machine Translation

This project implements a Transformer-based sequence-to-sequence model for neural machine translation, inspired by the original ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) paper. The code is modular, extensible, and uses PyTorch for all neural network components.

All core components—**multi-head self-attention**, **positional encoding**, and the **encoder-decoder structure**—are custom-built without relying on high-level implementations. The model is trained on the **Bilingual OPUS Books dataset** from Hugging Face, enabling multilingual translation using parallel text corpora. The project demonstrates how to construct and train a fully functional neural machine translation pipeline starting from raw data processing to generating translated text.

---

### Features

- **Custom Transformer Architecture**: Encoder-decoder structure with multi-head attention, positional encoding, and feed-forward layers.
- **Tokenization**: Uses the HuggingFace `tokenizers` library for building and loading word-level tokenizers.
- **Dataset**: Loads bilingual datasets from the HuggingFace `datasets` library (default: Helsinki-NLP/opus_books).
- **Training & Validation**: Full training loop with validation, checkpointing, and TensorBoard logging.
- **Masking**: Implements padding and causal masks for attention.
- **Configurable**: All hyperparameters and paths are set in a single config file.

---

### Project Structure

```
transformer/         # All model components (modules, encoder, decoder, etc.)
    modules.py
    encoder.py
    decoder.py
    build.py
    transformer.py
dataset.py           # Dataset class and masking utilities
train.py             # Training and validation pipeline
config.py            # Configuration and checkpoint utilities
```

---

### Setup

#### Install dependencies

You need Python 3.8+ and [PyTorch](https://pytorch.org/), plus the following packages:
```bash
pip install torch datasets tokenizers tqdm tensorboard
```

#### Download Data

The dataset is automatically downloaded using HuggingFace `datasets`.

#### Train the Model

```bash
python train.py
```

Training progress and validation examples will be printed and logged to TensorBoard.

---

### Configuration

Edit `config.py` or override values in your own script. Key options include:

- `batch_size`
- `num_epochs`
- `lr` (learning rate)
- `seq_len` (maximum sequence length)
- `d_model` (embedding/model dimension)
- `lang_src`, `lang_tgt` (source/target language codes)
- `model_folder`, `model_filename` (for checkpoints)
- `tokenizer_file` (tokenizer cache)
- `experiment_name` (TensorBoard log directory)

---

### Usage

- **Training:**  
  Run `python train.py` to start training from scratch or resume from a checkpoint.

- **Checkpoints:**  
  Model weights are saved every epoch in the `weights/` directory.

- **TensorBoard:**  
  Launch TensorBoard to monitor training:
  ```bash
  tensorboard --logdir runs/
  ```

---

### Customization

- **Change Dataset:**  
  Modify the dataset loading section in `train.py` to use a different HuggingFace dataset.

- **Model Architecture:**  
  Adjust the number of layers, heads, or dimensions in `config.py` and `build_transformer` in `transformer/build.py`.

---

### References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/)