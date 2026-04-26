import yaml
import os

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Create a sample config.yaml
sample_yaml = """
app:
  name: "Prediksi Saham 2.0"
  version: "2.0.0"

database:
  url: "sqlite:///./stock_predictions.db"

models:
  arima:
    order: [5, 1, 0]
  lstm:
    epochs: 50
    batch_size: 32
"""
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "w") as f:
    f.write(sample_yaml)
