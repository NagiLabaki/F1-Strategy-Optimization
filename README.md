## From Data to Decisions: Machine Learning for Formula One Race Strategy Optimization
# Team Members
Ameer Al Bitar – ama281@mail.aub.edu
Nagi Labaki – nsl03@mail.aub.edu
Nancy Fayad – nhf14@mail.aub.edu

Submitted to Dr. Joseph Bakarji – MECH 534

# Overview

This project applies machine learning to optimize Formula One race strategy by predicting lap times using different models. Our goal is to analyze these mdoels and try to improve them. Using real-world data from the Kaggle Formula 1 World Championship dataset and the FastF1 telemetry library, we build models that estimate lap times based on race, driver, and tire characteristics.

# Motivation

Formula 1 races rely heavily on strategic decisions that balance tire wear, fuel load, and driver pace. With teams increasingly integrating AI and ML into decision-making, our project explores how machine learning can improve lap-time forecasting, forming the foundation for automated strategy optimization.

# Datasets
1. Kaggle – Formula 1 World Championship (1950–2024)

Dataset Link: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020

This dataset includes historical F1 data such as races, results, lap times, pit stops, qualifying sessions, and constructors.

~589,000 lap-time records

14–18 features across linked tables

Covers 1950–2024 seasons

Ideal for large-scale trend and statistical modeling

2. FastF1 – FIA Telemetry API (2018–present)

FastF1 on GitHub

This dataset provides modern telemetry and weather information, including:

Tire compound and tire life

Track and air temperature

Stint and lap-by-lap data

Ideal for detailed analysis of tire degradation and environmental effects

These datasets can be merged using race and circuit identifiers to form a unified data source linking long-term performance data with modern telemetry.

We did some analysis and experimentation with the dataset which you can find in the jupyter notebooks and csv files in the main branch.

# Methods and Models

We implemented multiple supervised learning models to predict lap times as continuous values.

- Linear Regression (Baseline): interpretable model to estimate feature influence on lap time.

- Neural Networks: capture nonlinear dependencies between features (lap evolution, tire wear, etc.).

- Tree-Based Models (Random Forest): learn feature interactions and rank variable importance.

# Linear Regression

We used the following features in a linear regression model as a preliminary model:

- FuelLoadProxy: proportion of remaining fuel per lap.

- LapsBeforePit: average laps per tire stint.

- LapNumber²: captures nonlinear performance drop due to tire degradation.

This initial experiments achieved R² ≈ 0.67, capturing much of the lap-time variability while highlighting areas for model improvement. To improve predictive accuracy we extended the pipeline beyond linear regression and implemented both ensemble and neural approaches. A simple linear regression baseline (with engineered features such as laps-before-pit and a fuel-load proxy) performed poorly on absolute lap times (RMSE ≈ 7.9–8.6 s after cleaning and truncation), motivating non-linear models.

# Neural Networks

Neural Networks. We trained several NN variants. An absolute-time NN suffered from high variance on the raw target (MAE ≈ 1.63 s, RMSE ≈ 3.72 s reported for the initial absolute model), so we implemented a residual (baseline) formulation: the network predicts the deviation from a local baseline (average recent lap), which substantially reduced variance. The residual NN achieved MAE ≈ 1.16–1.19 s and an RMSE ≈ 3.30 s on the final test set; residual modelling tightened the error distribution around zero and improved robustness to outlier laps.

Telemetry-based NN. A telemetry-focused NN trained on FastF1 features (tyre life, fuel proxy, mean speed, throttle/brake, DRS, gear, and weather) and expanded from a two-race to a four-race dataset (Bahrain, Monaco, Silverstone, Monza) delivered the strongest telemetry-based results — reducing mean error into the sub-second range (the report documents the model improving to ≈0.83 s mean error as the dataset grew). This highlights the importance of diverse track data for generalization.

# Random Forest

A Random Forest regressor was used to capture non-linear feature interactions and proved robust to outliers. The forest achieved an MAE on the order of ~1.5 s (RMSE ≈ 2 s), and its feature ranking confirmed that recent lap history and median pace are strong predictors — consistent with a baseline-style signal.

# Short Takeaway

- The linear baseline establishes a low bar but fails to capture nonlinearity inherent in lap dynamics (large RMSE). 
- Tree ensembles provide strong, robust performance and clear feature importance (MAE ≈ 1.5 s). 
- Residual targets + neural networks produce the tightest error concentration and are most useful for fine-grained strategy work (residual MAE ≈ 1.16–1.19 s). 
- Adding telemetry from multiple circuits is the single most effective lever for improving generalization (final telemetry NN ≈ 0.83 s mean error).
