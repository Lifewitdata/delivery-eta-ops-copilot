"""Generate and execute the Delivery ETA analysis notebook."""

import os
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "01_eda_and_sla_baseline.ipynb"


def markdown(text: str) -> nbformat.v4.new_markdown_cell:
    return nbformat.v4.new_markdown_cell(text)


def code(text: str) -> nbformat.v4.new_code_cell:
    return nbformat.v4.new_code_cell(text)


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        markdown(
            "# Delivery ETA: EDA, chronological modeling, and SHAP\n\n"
            "This notebook analyzes a public delivery-operations dataset adapted to Saudi city names and zones. "
            "It is portfolio analysis, not real platform order data. The target is delivery time in minutes and the operational SLA assumption is 30 minutes."
        ),
        code(
            "from pathlib import Path\n"
            "import csv\n"
            "import gc\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.metrics import confusion_matrix\n"
            "from IPython.display import Markdown, display\n\n"
            "from src.features import engineer_features\n"
            "from src.train_model import train_models, SPLIT_DATE\n"
            "from src.explainability import build_summary\n\n"
            "ROOT = Path.cwd()\n"
            "DATA = ROOT / 'data'\n"
            "IMAGE_DIR = ROOT / 'docs' / 'images'\n"
            "IMAGE_DIR.mkdir(parents=True, exist_ok=True)\n"
            "sns.set_theme(style='whitegrid', context='notebook')\n"
            "plt.rcParams['figure.figsize'] = (10, 5)\n"
            "plt.rcParams['figure.dpi'] = 100\n"
            "pd.set_option('display.max_columns', 50)\n\n"
            "def csv_shape(path):\n"
            "    with path.open(encoding='utf-8-sig', newline='') as handle:\n"
            "        reader = csv.reader(handle)\n"
            "        header = next(reader)\n"
            "        rows = sum(1 for _ in reader)\n"
            "    return rows, len(header)"
        ),
        markdown("## 1. Data inventory and provenance"),
        code(
            "original_shape = csv_shape(DATA / 'raw' / 'Zomato Dataset.csv')\n"
            "cleaned_shape = csv_shape(DATA / 'raw' / 'zomato_cleaned.csv')\n"
            "orders = pd.read_csv(DATA / 'raw' / 'delivery_orders.csv')\n"
            "features = engineer_features(orders)\n"
            "inventory = pd.DataFrame({\n"
            "    'artifact': ['Kaggle original', 'Cleaned mirror', 'Adapted orders', 'Engineered features'],\n"
            "    'rows': [original_shape[0], cleaned_shape[0], len(orders), len(features)],\n"
            "    'columns': [original_shape[1], cleaned_shape[1], orders.shape[1], features.shape[1]],\n"
            "})\n"
            "display(inventory)\n"
            "gc.collect()\n"
            "display(Markdown(\n"
            "    f'**Date range:** {features.order_dt.min():%Y-%m-%d} to {features.order_dt.max():%Y-%m-%d}  \\n'\n"
            "    f'**Chronological split date:** {SPLIT_DATE:%Y-%m-%d}'\n"
            "))"
        ),
        markdown(
            "The original file is retained verbatim. The cleaned mirror has 38,964 rows. "
            "Saudi cities, coordinates, and order categories in the adapted file are synthetic assumptions and must not be interpreted as real platform behavior."
        ),
        markdown("## 2. Data quality"),
        code(
            "missing = orders.isna().sum()\n"
            "missing = missing[missing > 0].sort_values(ascending=True)\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "missing.plot.barh(ax=ax, color='#d95f02')\n"
            "ax.set_title('Missing values in adapted orders')\n"
            "ax.set_xlabel('Rows missing')\n"
            "fig.tight_layout()\n"
            "fig.savefig(IMAGE_DIR / 'data_quality.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()\n"
            "plt.close(fig)"
        ),
        markdown("## 3. Delivery-time and operations patterns"),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n"
            "sns.histplot(orders['delivery_time_min'], bins=30, kde=True, ax=axes[0], color='#2166ac')\n"
            "axes[0].axvline(30, color='#b2182b', linestyle='--', label='30-minute SLA')\n"
            "axes[0].legend()\n"
            "sns.boxplot(data=orders, x='saudi_city', y='delivery_time_min', ax=axes[1], color='#67a9cf')\n"
            "axes[1].tick_params(axis='x', rotation=20)\n"
            "fig.suptitle('Delivery time distribution and city comparison')\n"
            "fig.tight_layout()\n"
            "fig.savefig(IMAGE_DIR / 'delivery_patterns.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()\n"
            "plt.close(fig)"
        ),
        code(
            "fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))\n"
            "features.groupby('hour')['delivery_time_min'].mean().plot(ax=axes[0], marker='o')\n"
            "axes[0].set_title('Mean ETA by hour')\n"
            "orders.groupby('traffic')['delivery_time_min'].mean().sort_values().plot.bar(ax=axes[1], color='#ef8a62')\n"
            "axes[1].set_title('Mean ETA by traffic')\n"
            "orders.groupby('multiple_deliveries')['delivery_time_min'].mean().plot.bar(ax=axes[2], color='#7b3294')\n"
            "axes[2].set_title('Mean ETA by multiple deliveries')\n"
            "fig.tight_layout()\n"
            "fig.savefig(IMAGE_DIR / 'operations_patterns.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()\n"
            "plt.close(fig)"
        ),
        markdown("## 4. Chronological train/test split"),
        code(
            "daily = features.groupby(['order_date', features['order_dt'] < SPLIT_DATE]).size().unstack(fill_value=0)\n"
            "daily.columns = ['test', 'train']\n"
            "fig, ax = plt.subplots(figsize=(13, 4))\n"
            "daily['train'].plot(ax=ax, color='#2166ac', label='Train')\n"
            "daily['test'].plot(ax=ax, color='#b2182b', label='Test')\n"
            "ax.axvline(SPLIT_DATE, color='black', linestyle='--', label='Split date')\n"
            "ax.set_title('Daily order counts across the chronological split')\n"
            "ax.set_ylabel('Orders')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "fig.savefig(IMAGE_DIR / 'chronological_split.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()\n"
            "plt.close(fig)\n"
            "print('Train rows:', int((features.order_dt < SPLIT_DATE).sum()))\n"
            "print('Test rows:', int((features.order_dt >= SPLIT_DATE).sum()))"
        ),
        markdown("## 5. Model fitting and comparison"),
        code(
            "del orders\n"
            "gc.collect()\n"
            "metrics, test_predictions = train_models(features, ROOT)\n"
            "comparison = pd.DataFrame([metrics['baseline'], metrics['xgboost'], metrics['lightgbm']])\n"
            "display(comparison[['model', 'mae_min', 'rmse_min', 'r2', 'sla_accuracy', 'missed_late_count']])\n"
            "print('Winner:', metrics['winner'])"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "comparison.plot.bar(x='model', y='mae_min', ax=axes[0], color='#4393c3', legend=False)\n"
            "axes[0].set_title('Test MAE (lower is better)')\n"
            "axes[0].tick_params(axis='x', rotation=15)\n"
            "comparison.plot.bar(x='model', y='r2', ax=axes[1], color='#1b7837', legend=False)\n"
            "axes[1].set_title('Test R² (higher is better)')\n"
            "axes[1].tick_params(axis='x', rotation=15)\n"
            "fig.tight_layout()\n"
            "fig.savefig(IMAGE_DIR / 'model_comparison.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()\n"
            "plt.close(fig)"
        ),
        markdown("## 6. Prediction diagnostics and SLA errors"),
        code(
            "sample = test_predictions.sample(n=min(2500, len(test_predictions)), random_state=42)\n"
            "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n"
            "axes[0].scatter(sample['delivery_time_min'], sample['predicted_eta_min'], alpha=0.25, s=12)\n"
            "limits = [10, 55]\n"
            "axes[0].plot(limits, limits, '--', color='black')\n"
            "axes[0].set(xlim=limits, ylim=limits, xlabel='Actual ETA', ylabel='Predicted ETA', title='Actual vs predicted ETA')\n"
            "residuals = sample['delivery_time_min'] - sample['predicted_eta_min']\n"
            "sns.histplot(residuals, bins=35, kde=True, ax=axes[1], color='#d95f02')\n"
            "axes[1].axvline(0, color='black', linestyle='--')\n"
            "axes[1].set(title='Residual distribution', xlabel='Actual minus predicted (min)')\n"
            "fig.tight_layout()\n"
            "fig.savefig(IMAGE_DIR / 'prediction_diagnostics.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()\n"
            "plt.close(fig)"
        ),
        code(
            "cm = confusion_matrix(test_predictions['actual_sla_miss'], test_predictions['predicted_sla_miss'])\n"
            "fig, ax = plt.subplots(figsize=(5.5, 4.5))\n"
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['On time', 'Late'], yticklabels=['On time', 'Late'], ax=ax)\n"
            "ax.set(title='LightGBM SLA-call confusion matrix', xlabel='Predicted', ylabel='Actual')\n"
            "fig.tight_layout()\n"
            "fig.savefig(IMAGE_DIR / 'sla_confusion.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()\n"
            "plt.close(fig)"
        ),
        markdown("## 7. SHAP drivers and hour-band risk"),
        code(
            "summary = build_summary(metrics, test_predictions)\n"
            "(ROOT / 'models' / 'shap_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')\n"
            "drivers = pd.DataFrame(summary['top_shap_drivers']).sort_values('mean_abs_shap_min')\n"
            "fig, ax = plt.subplots(figsize=(9, 5))\n"
            "drivers.plot.barh(x='feature', y='mean_abs_shap_min', ax=ax, color='#2166ac', legend=False)\n"
            "ax.set(title='Global mean absolute SHAP values', xlabel='Mean |SHAP| (minutes)')\n"
            "fig.tight_layout()\n"
            "fig.savefig(IMAGE_DIR / 'shap_drivers.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()\n"
            "plt.close(fig)"
        ),
        code(
            "hour_bands = pd.DataFrame(summary['hour_band_summary'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "hour_bands.plot.bar(x='hour_band', y='sla_miss_rate', ax=ax, color='#b2182b', legend=False)\n"
            "ax.set(title='SLA miss rate by hour band', xlabel='Hour band', ylabel='SLA miss rate')\n"
            "ax.tick_params(axis='x', rotation=0)\n"
            "fig.tight_layout()\n"
            "fig.savefig(IMAGE_DIR / 'hour_band_risk.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()\n"
            "plt.close(fig)\n"
            "display(hour_bands[['hour_band', 'orders', 'avg_actual_eta', 'mae_min', 'sla_miss_rate']])"
        ),
        markdown("## 8. Findings and limits"),
        code(
            "winner_metrics = metrics[metrics['winner']]\n"
            "worst_zone = max(summary['zone_summary'], key=lambda item: item['sla_miss_rate'])\n"
            "display(Markdown(\n"
            "    f\"- **Model:** {metrics['winner'].title()} achieves **{winner_metrics['mae_min']:.2f} min MAE**, \"\n"
            "    f\"**{winner_metrics['r2']:.3f} R²**, and **{winner_metrics['sla_accuracy'] * 100:.1f}% SLA-call accuracy**.\\n\"\n"
            "    f\"- **Operational risk:** {worst_zone['zone']} has the highest zone miss rate at \"\n"
            "    f\"**{worst_zone['sla_miss_rate'] * 100:.1f}%**.\\n\"\n"
            "    f\"- **Time risk:** the evening band has the highest miss rate in the SHAP sample.\\n\"\n"
            "    '- **Limits:** source orders are from India in 2022; Saudi locations and categories are synthetic. '\n"
            "    'SHAP shows model reliance, not causation. Only 44 days are available, and missing order times fall back to midnight.'\n"
            "))"
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    executor = ExecutePreprocessor(timeout=600, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Executed notebook -> {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
