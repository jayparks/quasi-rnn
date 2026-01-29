# Product Requirements Document (PRD)
# Quasi-RNN Neural Machine Translation

**Version:** 1.0
**Date:** 2026-01-29
**Author:** Auto-generated from codebase analysis

---

## 1. Executive Summary

This project implements a Neural Machine Translation (NMT) system using Quasi-Recurrent Neural Networks (QRNNs) based on the paper "Quasi-Recurrent Neural Networks" (ICLR 2017). The system provides an encoder-decoder architecture with attention mechanism for sequence-to-sequence translation tasks, leveraging the computational efficiency benefits of QRNNs over traditional RNNs.

**Reference Paper:** https://arxiv.org/abs/1611.01576

---

## 2. Problem Statement

Traditional Recurrent Neural Networks (RNNs) and LSTMs, while effective for sequence modeling, suffer from:

- **Sequential computation bottleneck**: Each timestep must wait for the previous one to complete
- **Limited parallelization**: Cannot fully leverage GPU parallel processing capabilities
- **Training speed limitations**: Slow training times for long sequences

QRNNs address these issues by combining the parallelism of convolutional neural networks with the sequential modeling capabilities of recurrent networks, achieving up to 16x faster training speeds while maintaining competitive accuracy.

---

## 3. Goals and Objectives

### Primary Goals

1. **Implement QRNN-based NMT**: Provide a working implementation of encoder-decoder architecture using QRNN layers
2. **Enable efficient training**: Leverage QRNN's parallelizable convolution operations for faster training
3. **Support attention mechanism**: Implement attention for improved translation quality on long sequences

### Success Criteria

- Functional training pipeline with checkpoint support
- Working inference/decoding system
- Support for variable-length sequences
- Configurable model architecture (layers, hidden size, embedding size)

---

## 4. Target Users

| User Type | Use Case |
|-----------|----------|
| NLP Researchers | Experimenting with QRNN architectures for translation tasks |
| ML Engineers | Building production translation systems with efficient training |
| Students | Learning sequence-to-sequence models and attention mechanisms |
| Hobbyists | Training custom translation models on parallel corpora |

---

## 5. Core Features

### 5.1 QRNN Layer Implementation

| Feature | Description | Status |
|---------|-------------|--------|
| Convolutional Gates | Z (candidate), F (forget), O (output) gates via 1D convolution | Implemented |
| FO-Pooling | Forget-output pooling mechanism | Implemented |
| Attention Support | Optional attention on final decoder layer | Implemented |
| Variable Sequence Length | Masking for padded sequences | Implemented |

### 5.2 Model Architecture

| Component | Description | Status |
|-----------|-------------|--------|
| Encoder | Multi-layer QRNN with embedding | Implemented |
| Decoder | Multi-layer QRNN with attention | Implemented |
| Attention | Dot-product attention mechanism | Implemented |
| Output Projection | Linear layer to vocabulary logits | Implemented |

### 5.3 Training Pipeline

| Feature | Description | Status |
|---------|-------------|--------|
| Batch Training | Mini-batch gradient descent | Implemented |
| Checkpointing | Save/restore model state | Implemented |
| Validation Loop | Periodic validation with perplexity | Implemented |
| Gradient Clipping | Prevent exploding gradients | Implemented |
| Adam Optimizer | Adaptive learning rate optimization | Implemented |

### 5.4 Data Processing

| Feature | Description | Status |
|---------|-------------|--------|
| Vocabulary Builder | Create word-to-index dictionaries | Implemented |
| Bi-text Iterator | Load parallel corpora for training | Implemented |
| Batch Preparation | Padding, special tokens (_GO, _EOS) | Implemented |
| Corpus Shuffling | Shuffle while maintaining alignment | Implemented |
| Gzip Support | Read compressed data files | Implemented |

### 5.5 Inference/Decoding

| Feature | Description | Status |
|---------|-------------|--------|
| Greedy Decoding | Argmax token selection | Implemented |
| Beam Search | Top-k hypothesis tracking | Not Implemented |
| Batch Decoding | Parallel inference | Implemented |

---

## 6. Technical Architecture

### 6.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Data Loading │───▶│ Batch Prep   │───▶│ Model Forward│       │
│  │ (BiText)     │    │ (Padding)    │    │ Pass         │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                  │               │
│                                                  ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Checkpoint   │◀───│ Optimizer    │◀───│ Loss         │       │
│  │ Save         │    │ Step         │    │ Calculation  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Model Architecture

```
Source: "Hello world"          Target: "Bonjour monde"
         │                              │
         ▼                              ▼
┌─────────────────┐            ┌─────────────────┐
│ Source Embedding│            │ Target Embedding│
└────────┬────────┘            └────────┬────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐            ┌─────────────────┐
│ QRNN Layer 1    │            │ QRNN Layer 1    │
├─────────────────┤            ├─────────────────┤
│ QRNN Layer 2    │───────────▶│ QRNN Layer 2    │
│                 │  (memory)  │ + Attention     │
└────────┬────────┘            └────────┬────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐            ┌─────────────────┐
│ Encoder Output  │            │ Output Projection│
│ (Hidden States) │            │ (Vocabulary)    │
└─────────────────┘            └─────────────────┘
```

### 6.3 QRNN Layer Detail

```
Input Sequence: [x₁, x₂, x₃, ..., xₜ]
                        │
                        ▼
┌───────────────────────────────────────┐
│         1D Convolution                 │
│  (kernel_size × input_dim × 3×hidden) │
└───────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       Z (tanh)     F (sigmoid)   O (sigmoid)
          │             │             │
          └─────────────┼─────────────┘
                        ▼
┌───────────────────────────────────────┐
│         FO-Pooling (Sequential)        │
│  cₜ = (1-fₜ)⊙zₜ + fₜ⊙cₜ₋₁            │
│  hₜ = oₜ ⊙ cₜ                          │
└───────────────────────────────────────┘
                        │
                        ▼
Output: [h₁, h₂, h₃, ..., hₜ], [c₁, c₂, c₃, ..., cₜ]
```

---

## 7. Requirements

### 7.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | System shall train on parallel corpora (source-target pairs) | High |
| FR-02 | System shall support configurable model dimensions | High |
| FR-03 | System shall save/load model checkpoints | High |
| FR-04 | System shall handle variable-length sequences | High |
| FR-05 | System shall compute and report training perplexity | Medium |
| FR-06 | System shall perform validation during training | Medium |
| FR-07 | System shall decode/translate new source sentences | High |
| FR-08 | System shall build vocabularies from training data | High |

### 7.2 Non-Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-01 | Training should leverage GPU acceleration via PyTorch | High |
| NFR-02 | Memory usage should be configurable via batch size | Medium |
| NFR-03 | Code should be modular and maintainable | Medium |
| NFR-04 | System should handle gzip-compressed data files | Low |

### 7.3 Dependencies

| Package | Minimum Version | Purpose |
|---------|-----------------|---------|
| torch | >= 0.2.0 | Deep learning framework |
| numpy | >= 1.11.1 | Numerical computing |
| Python | 2.7+ | Runtime environment |

---

## 8. Current State and Known Issues

### 8.1 Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core QRNN Layer | Complete | Functional with attention support |
| Encoder-Decoder | Complete | Multi-layer support |
| Training Pipeline | Complete | With validation and checkpointing |
| Inference | Partial | Greedy decoding only |
| Data Pipeline | Complete | Full preprocessing support |

### 8.2 Known Issues / TODOs

| ID | Location | Issue | Severity |
|----|----------|-------|----------|
| TODO-01 | train.py:24 | Validate vocabulary size against actual dictionary | Low |
| TODO-02 | model.py:29 | Optimize masking with `masked_fill()` | Low |
| ISSUE-01 | All files | Uses deprecated Python 2.7 syntax | Medium |
| ISSUE-02 | All files | Uses deprecated PyTorch APIs (`Variable`, `.data[0]`) | Medium |

### 8.3 Technical Debt

1. **Python Version**: Code uses Python 2.7 syntax (`print` statements, `xrange`, `iteritems`)
2. **PyTorch API**: Uses deprecated patterns that may not work in PyTorch >= 1.0
3. **No Beam Search**: Decoding quality limited by greedy strategy
4. **No Unit Tests**: Missing test coverage

---

## 9. Future Improvements

### 9.1 Short-term (Next Release)

| Priority | Feature | Effort |
|----------|---------|--------|
| High | Migrate to Python 3 | Medium |
| High | Update to modern PyTorch API | Medium |
| Medium | Add beam search decoding | Medium |
| Medium | Add unit tests | Medium |

### 9.2 Medium-term

| Priority | Feature | Effort |
|----------|---------|--------|
| Medium | Add BPE/SentencePiece tokenization | Medium |
| Medium | Implement learning rate scheduling | Low |
| Low | Add TensorBoard logging | Low |
| Low | Docker containerization | Low |

### 9.3 Long-term

| Priority | Feature | Effort |
|----------|---------|--------|
| Low | Multi-GPU training support | High |
| Low | ONNX export for deployment | Medium |
| Low | REST API for inference | Medium |

---

## 10. Configuration Reference

### 10.1 Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--kernel_size` | 2 | QRNN convolution kernel size |
| `--hidden_size` | 1024 | Hidden layer dimension |
| `--emb_size` | 500 | Embedding dimension |
| `--num_layers` | 2 | Number of QRNN layers |
| `--num_enc_symbols` | 40000 | Source vocabulary size |
| `--num_dec_symbols` | 40000 | Target vocabulary size |
| `--batch_size` | 128 | Training batch size |
| `--max_epochs` | 10 | Maximum training epochs |
| `--lr` | 0.0002 | Learning rate |
| `--max_seq_len` | 50 | Maximum sequence length |
| `--clip_c` | 10.0 | Gradient clipping threshold |
| `--ckpt_frequency` | 2000 | Checkpoint save frequency |

### 10.2 Decoding Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 30 | Decoding batch size |
| `--max_decode_step` | 300 | Maximum decoding steps |

---

## 11. Usage Examples

### Build Vocabulary
```bash
python data/build_dictionary.py training_data.txt
# Output: training_data.txt.json
```

### Train Model
```bash
python train.py \
  --src_vocab source.json \
  --tgt_vocab target.json \
  --src_train source_train.txt \
  --tgt_train target_train.txt \
  --src_valid source_valid.txt \
  --tgt_valid target_valid.txt \
  --hidden_size 640 \
  --batch_size 128 \
  --model_dir checkpoints/ \
  --model_name qrnn_nmt.pkl
```

### Translate
```bash
python eval.py \
  -model_path checkpoints/qrnn_nmt.pkl \
  -src_vocab source.json \
  -tgt_vocab target.json \
  -decode_input test_source.txt \
  -decode_output translations.txt
```

---

## 12. Appendix

### A. File Structure

```
quasi-rnn/
├── README.md              # Project documentation
├── PRD.md                 # This document
├── layer.py               # QRNN layer implementation (65 lines)
├── model.py               # Encoder-decoder model (104 lines)
├── train.py               # Training script (225 lines)
├── eval.py                # Inference script (110 lines)
└── data/
    ├── __init__.py
    ├── data_utils.py      # Batch preparation (110 lines)
    ├── data_iterator.py   # Data loading (276 lines)
    ├── util.py            # File I/O utilities (55 lines)
    ├── build_dictionary.py # Vocabulary builder (42 lines)
    └── shuffle.py         # Corpus shuffling (59 lines)
```

### B. References

1. Bradbury, J., Merity, S., Xiong, C., & Socher, R. (2017). Quasi-Recurrent Neural Networks. ICLR 2017. https://arxiv.org/abs/1611.01576
2. Salesforce PyTorch QRNN: https://github.com/salesforce/pytorch-qrnn

---

*Document generated from automated codebase analysis.*
