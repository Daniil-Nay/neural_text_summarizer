# Neural Text Summarizer

A neural network implementation for contextual text summarization using the Transformer architecture. The project demonstrates high-quality text summarization capabilities with ROUGE-1 F1 scores of up to 0.85.

## Features
- Transformer-based architecture for text summarization
- Support for English language texts
- High-quality summaries with ROUGE metrics:
  - ROUGE-1 F1: 0.8547
  - ROUGE-2 F1: 0.7072
  - ROUGE-L F1: 0.8547
- Configurable model parameters
- Easy-to-use training and inference scripts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Daniil-Nay/neural_text_summarizer.git
cd neural_text_summarizer
```

2. Create and activate a virtual environment:
```bash
conda create -n summarizer_env python=3.8
conda activate summarizer_env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
- `src/` - source code
  - `model/` - transformer model implementation
    - `attention.py` - multi-head attention mechanism
    - `encoder.py` - transformer encoder
    - `decoder.py` - transformer decoder
    - `transformer.py` - full transformer model
    - `training.py` - training utilities
  - `data/` - data processing
    - `dataset.py` - dataset creation and loading
    - `preprocessing.py` - text preprocessing utilities
  - `utils/` - helper functions
    - `evaluation.py` - ROUGE metrics calculation

## Usage

### Training
To train the model with the best-performing parameters:
```bash
python src/train.py --epochs 100 --batch_size 2 --d_model 128 --num_heads 4 --dropout_rate 0.3 --max_length_input 500 --max_length_target 100
```

### Generating Summaries
To generate a summary for a new text:
```bash
python src/generate_summary.py --model_path models/transformer --text "Your text here"
```

## Results
The model achieves strong performance on test data:
- High ROUGE scores (ROUGE-1 F1: 0.8547)
- Coherent and accurate summaries
- Good handling of various topics (technology, science, business)

Example:
```
Input: "Stanford scientists have developed a revolutionary immunotherapy treatment that shows unprecedented success in early clinical trials. The treatment specifically targets aggressive forms of breast cancer by reprogramming the patient's own immune cells. Initial results show an exceptional response rate of over 70% in patients."

Generated Summary: "Stanford researchers groundbreaking immunotherapy treatment for breast cancer with 70% response rate."
```

## License
MIT

## Author
Daniil Nay 