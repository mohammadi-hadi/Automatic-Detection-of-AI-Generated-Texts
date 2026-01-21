<div align="center">

# Automatic Detection of AI-Generated Texts

### Multi-Model Ensemble for Detecting Machine-Generated Content

[![CLIN Journal](https://img.shields.io/badge/CLIN%20Journal-Article%20182-blue.svg)](https://clinjournal.org/clinj/article/view/182)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.0+-orange.svg)](https://huggingface.co/transformers/)

*Ensemble approach combining BERT, XLM-RoBERTa, and DistilBERT for robust AI text detection*

---

</div>

## Paper

**Title:** Automatic Detection of AI-Generated Texts

**Authors:** Hadi Mohammadi, Anastasia Giachanou, Ayoub Bagheri

**Published in:** Computational Linguistics in the Netherlands Journal (CLIN)

**Link:** [https://clinjournal.org/clinj/article/view/182](https://clinjournal.org/clinj/article/view/182)

## Overview

This repository contains an ensemble model for detecting AI-generated texts. The approach combines three transformer-based models (BERT, XLM-RoBERTa, and DistilBERT) with a voting mechanism to achieve robust classification across multiple languages and domains.

## Key Features

- **Multi-Model Ensemble**: Combines predictions from BERT, XLM-RoBERTa, and DistilBERT
- **Multilingual Support**: Handles both English and Dutch text
- **Domain Adaptability**: Tested on news, reviews, and Twitter data
- **Voting Mechanism**: Ensemble voting for improved robustness

## Results

| Dataset | Accuracy | F1 Score |
|---------|----------|----------|
| English News | 97.50% | 97.50% |
| English Twitter | 96.00% | 96.00% |
| Dutch News | 97.00% | 97.00% |
| Dutch Twitter | 92.50% | 92.47% |
| English Reviews | 81.50% | 81.21% |
| Dutch Reviews | 84.00% | 84.00% |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mohammadi-hadi/Automatic-Detection-of-AI-Generated-Texts.git
cd Automatic-Detection-of-AI-Generated-Texts

# Install dependencies
pip install torch transformers pandas scikit-learn

# Open the notebook
jupyter notebook contestant_model.ipynb
```

## Model Architecture

```python
class CustomBERTModel(nn.Module):
    """
    Ensemble model combining:
    - bert-base-multilingual-cased
    - xlm-roberta-base
    - distilbert-base-multilingual-cased

    Uses voting mechanism for final prediction
    """
```

## Usage

```python
from transformers import BertTokenizer
import torch

# Load model
model = torch.load('entire_custom_bert_model.pth')
model.eval()

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

# Predict
def predict(model, tokenizer, text, max_length=256):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding='max_length', max_length=max_length)
    with torch.no_grad():
        logits = model(inputs['input_ids'])
        final_logits, _ = torch.mode(logits, dim=0)
        preds = torch.argmax(final_logits, dim=1)
    return preds.item()

# Example
result = predict(model, tokenizer, "Your text here")
print(f"Prediction: {'AI-generated' if result == 1 else 'Human-written'}")
```

## Repository Structure

```
Automatic-Detection-of-AI-Generated-Texts/
├── contestant_model.ipynb  # Main notebook with model and evaluation
├── LICENSE                 # MIT License
├── CONTRIBUTING.md         # Contribution guidelines
└── README.md              # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.0+
- pandas
- scikit-learn

## Citation

```bibtex
@article{mohammadi2024aigentexts,
  title={Automatic Detection of AI-Generated Texts},
  author={Mohammadi, Hadi and Giachanou, Anastasia and Bagheri, Ayoub},
  journal={Computational Linguistics in the Netherlands Journal},
  year={2024},
  url={https://clinjournal.org/clinj/article/view/182}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration inquiries:
- **Hadi Mohammadi** - Utrecht University
- **Email**: [h.mohammadi@uu.nl](mailto:h.mohammadi@uu.nl)
- **Website**: [mohammadi.cv](https://mohammadi.cv)
