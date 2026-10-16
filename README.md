# Data_science

Crop Recommendation System is a data science project that trains Decision Tree
and Random Forest models to recommend crops based on soil nutrients, pH,
temperature, humidity, and rainfall.

## Run locally

Install dependencies and run the script from the project directory:

```bash
python -m pip install -r requirements.txt
python croup.py
```

The script loads `Crop_recommendation.csv` by default. You can provide another
CSV with `--data`. To make a prediction without interactive prompts, pass the
seven feature values in this order: nitrogen, phosphorus, potassium,
temperature, humidity, pH, and rainfall:

```bash
python croup.py --predict 90 42 43 20.88 82 6.5 202.9
```

## Run on GitHub

GitHub Actions installs the dependencies, checks the script, then trains and
evaluates the models on pushes and pull requests targeting `main`. You can also
start it manually from the repository's **Actions** tab. Interactive prediction
is skipped unless values are supplied with `--predict`.
