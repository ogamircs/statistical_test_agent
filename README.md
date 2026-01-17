# Statistical Test Agent

An AI-powered A/B Testing Agent built with Python and Streamlit. This tool automates the process of analyzing A/B test results by intelligently selecting the appropriate statistical methods, checking for bias/imbalance, and providing both Frequentist and Bayesian interpretations.

## Table of Contents
- [Installation & Setup](#installation--setup)
- [Running the Agent](#running-the-agent)
- [Project Structure](#project-structure)
- [How it Works (Methodology)](#how-it-works-methodology)

## Installation & Setup

We recommend using `uv` (a fast Python package installer and resolver) to manage the environment.

### 1. Install `uv`
If you haven't installed `uv` yet, you can do so following the [official instructions](https://github.com/astral-sh/uv).
On Windows (PowerShell):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Create a Virtual Environment
Navigate to the project directory and create a new virtual environment:

```powershell
uv venv
```

Activate the environment:
- **Windows**: `.\.venv\Scripts\activate`
- **Mac/Linux**: `source .venv/bin/activate`

### 3. Install Dependencies
Install the required packages (Streamlit, Pandas, NumPy, Plotly, SciPy, Scikit-learn):

```powershell
uv pip install streamlit pandas numpy plotly scipy scikit-learn
```

---

## Running the Agent

To start the user interface, run the following command from the project root:

```powershell
.\.venv\Scripts\streamlit run src/app.py
```

This will open the application in your default web browser (usually at `http://localhost:8501`).

### Quick Start
1.  **Upload Data**: Drag and drop your CSV file into the sidebar.
    -   You can use the sample data provided in `data/sample_ab_data.csv` to test it out.
2.  **Configuration**: The agent will automatically guess your columns (ID, Group, Metric, Segment, etc.). You can verify and adjust these in the "Data Configuration" section.
3.  **View Results**: Navigate through the tabs to see:
    -   **Frequentist Results**: Standard T-Tests or Diff-in-Diff results.
    -   **Bayesian Results**: Probability distributions and "Probability Treatment > Control".
    -   **Segment Analysis**: Breakdown of effects by user segment.
    -   **Overall Comparison**: A head-to-head view of Frequentist vs Bayesian conclusions.

---

## Project Structure

```text
statistical_test_agent/
├── data/               # Sample CSV files for testing
├── src/
│   ├── ab_agent.py     # Core logic (ABAgent class) handling stats & analysis
│   ├── app.py          # Streamlit user interface implementation
├── tests/              # Unit tests
└── README.md           # This documentation
```

---

## How it Works (Methodology)

The Agent (`ABAgent` class in `src/ab_agent.py`) follows a rigorous statistical workflow:

### 1. Smart Column Detection
The agent uses heuristics to automatically identify identifying columns:
-   **ID**: Looks for `user_id`, `id`, `cust_id`.
-   **Group**: Looks for `group`, `variant`, or columns with 2-3 unique values (e.g., "control", "test).
-   **Metric**: Looks for continuous outcome variables like `revenue`, `conversion`, `amount`.
-   **Pre-Experiment Metric**: Checks for columns like `pre_revenue` to enable Difference-in-Differences (DiD).

### 2. Balance Checks & Bias Correction
Before analyzing, the agent checks validity:
-   **Covariate Imbalance**: Checks if user features (age, region, etc.) are balanced across groups using **Standardized Mean Differences (SMD)** and **Chi-Square** tests.
    -   *Correction*: If imbalance is found, it automatically applies **Inverse Probability Weighting (IPW)** using Logistic Regression to re-weight the data.
-   **Pre-Experiment Balance (A/A Test)**: If a pre-experiment metric is available (e.g., revenue *before* the test), it runs a T-Test on it.
    -   *Correction*: If the A/A test fails ($p < 0.05$), the agent automatically switches from a standard T-Test to **Difference-in-Differences (DiD)** to control for pre-existing bias.

### 3. Frequentist Analysis
-   **Method**: Two-sample T-Test (Welch's t-test) or DiD.
-   **Stratification**: If a segment is selected, it calculates the effect size within each segment and aggregates them using a **Weighted Average** (weighted by sample size).
    -   $Overall Effect = \frac{\sum (Effect_i \times Weight_i)}{\sum Weight_i}$

### 4. Bayesian Analysis
-   **Priors**: Uses objective Conjugate Priors.
    -   **Binary Metrics**: Beta-Bernoulli model.
    -   **Continuous Metrics**: Normal-Inverse-Gamma assumption (approximated via T-distribution of the mean).
-   **Outputs**:
    -   **Posterior Distributions**: Visualizes the likely range of the true mean/rate.
    -   **Probability Analysis**: Calculates $P(Treatment > Control)$ and Expected Uplift.
    -   **Stratified Bayesian**: Simulates samples from each segment's posterior and aggregates them to form a "Total Weighted Posterior," giving a unified view of the experiment while respecting segment differences.
