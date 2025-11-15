# Customer Lifetime Value (CLV) Prediction with AFT Models

This homework demonstrates how to compute Customer Lifetime Value (CLV) using parametric survival models (Accelerated Failure Time models) from the `lifelines` library. 

The workflow includes:

- Loading and exploring customer churn data
- Preprocessing (handling missing values, encoding categorical variables)
- Fitting Weibull, LogNormal, and LogLogistic AFT models
- Comparing models using AIC and concordance index
- Backward feature elimination based on p-values
- Predicting survival curves and expected CLV per customer
- Saving results and exploring CLV across segments

## Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

## Usage

Run:

```
python main.py
```

Results will be saved in the `results/` directory, including `clv_results.csv` with predicted CLV for each customer.