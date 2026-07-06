# Titanic Feature Engineering: Predictive Modeling with Function Transformers

This repository demonstrates the practical application of **Mathematical Function Transformations** as a crucial feature engineering step in machine learning pipelines. Using the classic Titanic dataset, this project explores how transforming skewed numerical data can significantly impact the performance of linear and tree-based classification models.

## Project Overview

In machine learning, algorithms like Logistic Regression assume that numerical features are normally distributed. Highly skewed variables (such as passenger `Fare` or `Age`) can degrade model performance. 

This project implements `scikit-learn`'s `FunctionTransformer` and `ColumnTransformer` to apply mathematical functions (e.g., Logarithmic, Square Root, Reciprocal) to features, aiming to reduce skewness and improve predictive accuracy.

### Key Features Analyzed:
*   **Survived:** Target variable (0 = No, 1 = Yes).
*   **Age:** Numerical feature with missing entries (177 missing values identified).
*   **Fare:** Numerical feature typically exhibiting high right-skewness.

## Tech Stack & Libraries

*   **Language:** Python 3
*   **Data Manipulation:** `pandas`, `numpy`
*   **Statistical Analysis & Visualization:** `scipy.stats`, `matplotlib`, `seaborn`
*   **Machine Learning Pipeline:** `scikit-learn`
    *   *Models:* `LogisticRegression`, `DecisionTreeClassifier`
    *   *Preprocessing:* `FunctionTransformer`, `ColumnTransformer`
    *   *Evaluation:* `train_test_split`, `cross_val_score`, `accuracy_score`


##  Pipeline Workflow

1. **Data Loading & Exploration:** Filtered dataset to isolate key numerical predictors and checked for missing data profiles.
2. **Distribution Analysis:** (In Progress) Checking skewness using Q-Q plots and histograms.
3. **Transformation Application:** Utilizing `ColumnTransformer` to selectively apply custom mathematical mappings to specific features without leaking data.
4. **Model Training & Evaluation:** Comparing baseline model accuracy against post-transformation model accuracy using Stratified Cross-Validation.

##  Key Architectural Insights

### 1. The Skewness Profile (Age vs. Fare)
*   **The `Fare` Feature:** In the Titanic dataset, passenger fare typically follows a highly right-skewed, power-law-like distribution. A few passengers paid extremely high prices for first-class suites, while the majority paid very low fares. Linear models like `LogisticRegression` will struggle with this scale discrepancy. A **Logarithmic transformation** or a **Box-Cox/Yeo-Johnson transformation** will be essential here to compress the variance and stabilize the distribution.
*   **The `Age` Feature:** Age usually follows a much closer-to-normal distribution, though slightly right-skewed or bimodal (due to infants). It might benefit more from a **Square Root** transformation or potentially no transformation at all depending on your QQ-plots.

### 2. The `NaN` Roadblock in `Age`
*   Your output `df.isnull().sum()` reveals **177 missing values** in `Age`. 
*   **Pipeline Constraint:** `FunctionTransformer` mappings (like `np.log` or `np.sqrt`) will propagate or fail when encountering missing values. You **cannot** pass this raw data directly into your `ColumnTransformer` for mathematical functions without handling these gaps first. You must inject an imputation step (e.g., filling missing values with the median age) using `SimpleImputer` prior to running the function transformations.

### 3. Model Sensitivity Differences
*   **`LogisticRegression`:** This model relies on gradient descent/coordinate descent and assumes a linear relationship with log-odds. It is highly sensitive to feature scaling and skewness. You should expect to see a noticeable boost in cross-validation accuracy for this model once your transformations are applied.
*   **`DecisionTreeClassifier`:** Tree-based models split data based on monotonic thresholds (e.g., $Age > 28$). Because mathematical functions like log or square root are monotonic transformations, they do not change the relative ordering of the data. Therefore, the transformations will likely have **zero or negligible impact** on the Decision Tree's performance, providing a great baseline to contrast against your linear model.
