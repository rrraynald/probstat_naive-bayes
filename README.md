# Naive Bayes to Detect Cyber Attack

Team 5:

1. Ferdian Ardra Hafizhan (5025241033)
2. Raynald Ramadhani Fachriansyah (5025241020)
3. Gilbran Mahdavikia Raja (5025241134)
4. Afarrel Febryan Ghiffari Putra Andy (5025241137)

## Bayes Theory

## How Bayes Theory Implemented on Naive Bayes Model

## Overview about the Dataset

![alt text](asset/image.png)

The dataset used in this project contains simulated cybersecurity session data that represents network activities and potential attack patterns.

link to dataset: https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset

Each row represents a network session and contains several features related to network behavior.

### Dataset Features

| Feature | Type | Description |
|---|---|---|
| network_packet_size | Numerical | The size of packets transmitted during a network session. |
| protocol_type | Categorical | The network protocol used during communication (TCP or UDP). |
| login_attempts | Numerical | Number of login attempts within the session. |
| session_duration | Numerical | The total duration of the network session. |
| encryption_used | Categorical | Type of encryption used in the connection. |
| ip_reputation_score | Numerical | A score indicating the trustworthiness of the source IP address. |
| failed_logins | Numerical | Number of failed authentication attempts. |
| browser_type | Categorical | Browser used in the session. |
| unusual_time_access | Binary | Indicates whether the session occurred at an unusual time. |
| attack_detected | Binary | Target label indicating whether the session is classified as a cyber attack. |

Target label indicating whether the session is classified as a cyber attack.

### Dataset 5 First Row:

| session_id | network_packet_size | protocol_type | login_attempts | session_duration | encryption_used | ip_reputation_score | failed_logins | browser_type | unusual_time_access | attack_detected |
|---|---|---|---|---|---|---|---|---|---|---|
| SID_00001 | 599 | TCP | 4 | 492.9832634426563 | DES | 0.606818080396889 | 1 | Edge | 0 | <span style="color:red;font-weight:bold">1</span> |
| SID_00002 | 472 | TCP | 3 | 1557.9964611204384 | DES | 0.30156896759608937 | 0 | Firefox | 0 | <span style="color:green;font-weight:bold">0</span> |
| SID_00003 | 629 | TCP | 3 | 75.04426166420741 | DES | 0.7391643279163831 | 2 | Chrome | 0 | <span style="color:red;font-weight:bold">1</span> |
| SID_00004 | 804 | UDP | 4 | 601.2488351708328 | DES | 0.12326717575248465 | 0 | Unknown | 0 | <span style="color:red;font-weight:bold">1</span> |
| SID_00005 | 453 | TCP | 5 | 532.5408884201419 | AES | 0.05487385674317035 | 1 | Firefox | 0 | <span style="color:green;font-weight:bold">0</span> |

## Naive Bayes Implementation for Cyber Attack Detection

This project implements several variants of the Naive Bayes algorithm to detect cyber attacks from network session data. The full pipeline consists of data exploration, preprocessing, model training, hyperparameter tuning, and evaluation.

The workflow follows a standard machine learning pipeline to ensure reliable and reproducible results.

---

### 1. Data Loading

The dataset is loaded using the pandas library and contains multiple features representing network session behavior.

```python
df = pd.read_csv("./cybersecurity_intrusion_data.csv")
```
Each row represents a single network session.

### 2. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the dataset structure and identify patterns related to cyber attacks.

Several visualizations were used to explore the data:

- **Attack vs Normal Distribution** to observe class balance.
- **Protocol Type vs Attack** to analyze whether certain network protocols are more associated with attacks.
- **IP Reputation Score vs Attack** to examine whether suspicious IP addresses correlate with malicious sessions.
- **Correlation Heatmap** to identify relationships between numerical features.

These analyses help provide an initial understanding of the dataset before training machine learning models.

### 3. Data Preprocessing

Before applying the Naive Bayes models, several preprocessing steps were performed.

- #### Removing Irrelevant Features

- #### Handling Missing Values

    Missing values in the `encryption_used` feature were replaced with the category `"Unknown"`.

- #### Encoding Categorical Features

    Since machine learning models require numerical input, categorical features were converted into numerical values using Label Encoding.

After preprocessing, the dataset is ready to be used for training Naive Bayes models.

#### Feature and Target Separation

The dataset is split into input features and the target variable.

```python
X = df.drop("attack_detected", axis=1)
y = df["attack_detected"]
```

- **X** contains all input features  
- **y** represents the target label indicating whether the session is a cyber attack

### 5. Cross Validation using Stratified K-Fold

Instead of using a single train test split, the project uses Stratified K-Fold cross validation.

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Stratified K-Fold ensures that each fold maintains the same class distribution as the original dataset.

### 6. Naive Bayes Models

Several variants of the Naive Bayes algorithm are implemented and compared in this project.

```python
models = {
    "Gaussian NB": GaussianNB(),
    "Multinomial NB": MultinomialNB(),
    "Bernoulli NB": BernoulliNB(),
    "Complement NB": ComplementNB()
}
```

Naive Bayes is a probabilistic classification algorithm based on **Bayes' Theorem** with a strong assumption that all features are **conditionally independent given the class label**.

#### Bayes Theorem

The general formula of Bayes' Theorem is:

```
P(C | X) = (P(X | C) * P(C)) / P(X)
```

Where:

- `C` = class label (Attack or Normal)
- `X` = feature vector
- `P(C | X)` = posterior probability
- `P(X | C)` = likelihood
- `P(C)` = prior probability
- `P(X)` = evidence

Since `P(X)` is constant for all classes, the equation becomes:

```
P(C | X) ∝ P(C) * Π P(x_i | C)
```

The classifier predicts the class with the highest probability:

```
Ĉ = argmax P(C) * Π P(x_i | C)
```

Different Naive Bayes variants assume different probability distributions for `P(x_i | C)`.

---

### Gaussian Naive Bayes

Gaussian Naive Bayes assumes that continuous features follow a **normal distribution**.

This model is suitable for numerical features such as:

- network_packet_size  
- session_duration  
- ip_reputation_score  

The likelihood of each feature is calculated using the Gaussian probability density function:

```
P(x_i | C) =
1 / sqrt(2πσ²) * exp(-(x_i - μ)² / (2σ²))
```

Where:

- `μ` = mean of feature for class C
- `σ²` = variance of feature for class C

This allows the model to estimate probabilities for continuous variables.

### Multinomial Naive Bayes

Multinomial Naive Bayes is commonly used for **discrete count features**, such as word frequencies in text classification.

The probability is estimated using:

```
P(x_i | C) =
(N_ic + α) / (N_c + αn)
```

Where:

- `N_ic` = number of occurrences of feature i in class C
- `N_c` = total feature count in class C
- `n` = number of features
- `α` = smoothing parameter (Laplace smoothing)

Laplace smoothing prevents zero probabilities when a feature does not appear in a class.

### Bernoulli Naive Bayes

Bernoulli Naive Bayes is designed for **binary features**.

Each feature is treated as a binary variable indicating presence or absence.

The likelihood is calculated as:

```
P(x_i | C) =
p_ic^(x_i) * (1 - p_ic)^(1 - x_i)
```

Where:

- `x_i` ∈ {0,1}
- `p_ic` = probability that feature i appears in class C

This model is effective when features represent binary indicators.

### Complement Naive Bayes

Complement Naive Bayes is an improved variant of Multinomial Naive Bayes designed to handle **imbalanced datasets**.

Instead of calculating probabilities from the target class directly, it uses the **complement of the class**.

The weight calculation is:

```
w_ci = log((N_ci + α) / (Σ N_cj + αn))
```

Where:

- `N_ci` = count of feature i in complement class
- `n` = number of features
- `α` = smoothing parameter

This approach reduces bias toward majority classes.

### Summary of Naive Bayes Variants

| Model | Feature Assumption | Suitable Data |
|------|------|------|
| Gaussian NB | Continuous Gaussian distribution | Numerical features |
| Multinomial NB | Multinomial distribution | Count-based features |
| Bernoulli NB | Bernoulli distribution | Binary features |
| Complement NB | Multinomial with complement weighting | Imbalanced datasets |

In this project, these variants are compared to determine which Naive Bayes model performs best for detecting cyber attacks in network session data.

### 7. Model Training

Each Naive Bayes model is trained using Stratified K-Fold cross validation.

For each fold:

1. Training data is used to fit the model  
2. The model predicts labels on the validation fold  
3. Evaluation metrics are computed

Example training step:

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

This process is repeated for all folds to obtain stable performance metrics.

### 8. Model Evaluation

![alt text](asset/image-3.png)

Model performance is evaluated using several classification metrics derived from the **confusion matrix**.

| Term | Description |
|-----|------|
| **True Positive (TP)** | Attack sessions correctly classified as attacks |
| **True Negative (TN)** | Normal sessions correctly classified as normal |
| **False Positive (FP)** | Normal sessions incorrectly classified as attacks |
| **False Negative (FN)** | Attack sessions incorrectly classified as normal |

#### Accuracy

Accuracy measures the overall correctness of the model.

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
It represents the proportion of correctly classified instances among all predictions.

#### Precision

Precision measures how many predicted attacks are actually attacks.

```
Precision = TP / (TP + FP)
```
High precision means the model produces fewer **false alarms**.

#### Recall

Recall measures how many real attacks were correctly detected.

```
Recall = TP / (TP + FN)
```

High recall indicates that the model successfully detects most attacks.

#### F1 Score

The F1 score is the harmonic mean of precision and recall.

```
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
```

It can also be written using confusion matrix values:

```
F1 Score = 2TP / (2TP + FP + FN)
```

The F1 score is particularly useful when dealing with **imbalanced datasets**, as it balances precision and recall.

The results from each fold are averaged to produce final metrics.

### 9. Confusion Matrix

Confusion matrices are used to visualize model prediction performance.

Heatmaps are used to visualize the confusion matrix for each model.

![alt text](asset/image-1.png)
![alt text](asset/image-2.png)

### 10. Hyperparameter Tuning

Hyperparameter tuning is applied using GridSearchCV to find optimal parameters for Naive Bayes models.

Example tuning for Gaussian Naive Bayes:

```python
param_grid = {
    "var_smoothing": np.logspace(-12, -3, 10)
}

grid_search = GridSearchCV(
    estimator=GaussianNB(),
    param_grid=param_grid,
    cv=skf,
    scoring="accuracy",
    n_jobs=-1
)
```
#### Model Accuracy after Hyperparameter Tuning

![alt text](asset/image-4.png)

The best model configuration is selected based on cross validation accuracy.

### 11. Compare to other models

To evaluate the effectiveness of Naive Bayes, the results are compared with several other machine learning algorithms.

These include:

- Logistic Regression  
- K-Nearest Neighbors  
- Decision Tree  
- Random Forest  
- Support Vector Machine  
- Boosting models such as XGBoost, LightGBM, and CatBoost

The performance of each model is compared using bar chart visualizations.

![alt text](asset/image-5.png)

The performance of Naive Bayes models was compared with several other machine learning algorithms, including tree-based models, linear models, and gradient boosting models.

Based on the evaluation results, boosting algorithms such as **CatBoost, Random Forest, LightGBM, and XGBoost** achieved the highest accuracy. These models are known for their strong performance on tabular datasets because they can capture complex feature interactions and nonlinear relationships.

Among the Naive Bayes variants, **Gaussian Naive Bayes** achieved the best performance. This is expected because most features in the dataset are continuous numerical variables such as packet size, session duration, and IP reputation score, which align well with the Gaussian distribution assumption.

Other Naive Bayes variants such as **Multinomial, Bernoulli, and Complement Naive Bayes** performed lower in comparison. These models are typically more suitable for text classification or binary feature representations rather than continuous tabular data.

Despite having lower accuracy compared to boosting models, Naive Bayes still provides several advantages:

- Very fast training and prediction
- Simple probabilistic interpretation
- Strong baseline model for classification tasks

Therefore, Naive Bayes can serve as an efficient baseline model for cyber attack detection, while more complex models such as boosting algorithms can be used to achieve higher predictive performance.