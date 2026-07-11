# Open Source Price Predictor

A product price prediction project that fine-tunes an open-source model to estimate product prices from product descriptions.

## Files


- `prompt_data_and_base_model.ipynb` - prompt data preparation and base model workflow.
- `open_source_model_training.ipynb` - open-source model fine-tuning workflow.
- `fine_tuned_model_evaluation.ipynb` - evaluation workflow for the fine-tuned model.
- `results_analysis.ipynb` - results notebook.
- `pricer/` - reusable pricing/evaluation Python package.
- `util.py` - utility helpers used by the notebooks.
- `requirements.txt` - Python dependencies from the course environment.

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

Training notebooks are designed for GPU environments such as Google Colab. Add any required Hugging Face or model provider credentials before running training/evaluation cells.
