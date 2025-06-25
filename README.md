# E-commerce Data Analysis & NER Project

This project focuses on analyzing e-commerce data from Telegram channels, with a strong emphasis on Natural Language Processing (NLP) for Named Entity Recognition (NER) and vendor analytics.

## Project Structure

- **scripts/**: Data preprocessing, NER labeling, Telegram scraping, utilities, and vendor scorecard generation scripts.
- **notebooks/**: Jupyter notebooks for NER fine-tuning, model comparison, interpretability (SHAP), and vendor analytics visualizations.
- **tests/**: Placeholder for test scripts.

## Main Features

- **Data Collection & Preprocessing**: Scrape and clean Telegram data, label it for NER tasks.
- **NER Model Training & Evaluation**: Fine-tune transformer models, evaluate and compare using classification reports.
- **Model Interpretation**: Use SHAP to interpret model predictions.
- **Vendor Analytics & Visualization**: Analyze and visualize vendor performance metrics (message counts, lending scores, price distributions).

## Technologies Used

- Python (transformers, datasets, seqeval, pandas, matplotlib, seaborn, plotly, SHAP)
- Jupyter Notebooks

## Getting Started

1. Clone the repository and install required Python packages (see notebook cells for package lists).
2. Use scripts in `scripts/` for data preprocessing and NER labeling.
3. Run and modify notebooks in `notebooks/` for model training, evaluation, interpretation, and analytics.

## Example Notebooks

- `ner_fine_tuning.ipynb`: Fine-tune and evaluate NER models.
- `model_comparison.ipynb`: Compare different NER models.
- `model_interpretability.ipynb`: Interpret model predictions with SHAP.
- `vendor_scorecard_visuals.ipynb`: Visualize vendor analytics and scorecards.

## License

This project is for educational and research purposes.
