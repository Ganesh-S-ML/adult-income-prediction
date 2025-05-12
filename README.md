## Adult Income Prediction

This project predicts whether an individual's income exceeds $50K per year based on census data using machine learning models.

## Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult)
- **Data URL:** [adult.data](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)

## ⚙️ Setup
```bash
# Clone the repository
git clone <repository-url>
cd adult-income-prediction

# Create a virtual environment
# For Windows:
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

#Required Libraries
The following libraries are used in this project:

pandas
numpy
matplotlib
seaborn
scikit-learn
shap
joblib


 

## How to run
1. Run the Jupyter Notebook
Open the adult_income.ipynb file in Jupyter Notebook or any compatible IDE (e.g., VS Code with Jupyter extension). Follow the cells sequentially to:

Load and preprocess the data.
Train machine learning models (Logistic Regression and Random Forest).
Evaluate model performance.
Interpret results using SHAP and permutation importance.

2. Key Outputs
Model Performance Metrics:
Accuracy, ROC AUC, and F1 Score for both Logistic Regression and Random Forest models.
Feature Importance:
Permutation importance and SHAP analysis to identify the top predictors of income.
Visualizations:
Boxplots and barplots for exploratory data analysis (EDA).

3. Predict New Data
To predict income for new data:

Ensure the pipeline is serialized (random_forest_pipeline.pkl).
Use the following code

# Load the pipeline
loaded_pipeline = joblib.load("random_forest_pipeline.pkl")
print("Pipeline loaded successfully.")

# Define the new data
new_data = pd.DataFrame({
    "age": [39],
    "workclass": ["Private"],
    "fnlwgt": [123456],
    "education": ["Bachelors"],
    "education-num": [13],
    "marital-status": ["Married-civ-spouse"],
    "occupation": ["Exec-managerial"],
    "relationship": ["Husband"],
    "race": ["White"],
    "gender": ["Male"],
    "capital-gain": [0],
    "capital-loss": [0],
    "hours-per-week": [40],
    "native-country": ["United-States"]
})

# Define the age_group function
def age_group(x):
    x = int(x)
    x = abs(x)
    if 18 < x < 31:
        return "19-30"
    if 30 < x < 41:
        return "31-40"
    if 40 < x < 51:
        return "41-50"
    if 50 < x < 61:
        return "51-60"
    if 60 < x < 71:
        return "61-70"
    else:
        return "Greater than 70"

# Add the age_group column to the new data
new_data["age_group"] = new_data["age"].apply(age_group)

# Add feature engineering steps
new_data["education_level_per_hour"] = new_data["education-num"] * new_data["hours-per-week"]
new_data["is_capital_gain"] = (new_data["capital-gain"] > 0).astype(int)

# Preprocess new data
new_data_processed = loaded_pipeline.named_steps["preprocessor"].transform(new_data)

# Predict income class
prediction = loaded_pipeline.predict(new_data)
print("Prediction:", ">50K" if prediction[0] == 1 else "<=50K")


## Interpreting Results
1. Model Performance
Logistic Regression:
Accuracy: ~86%
ROC AUC: ~0.90
Random Forest:
Accuracy: ~83%
ROC AUC: ~0.88
2. Key Predictors of Income
Top Features (SHAP Analysis):
capital-gain: Strongest predictor of high income.
marital-status_Married-civ-spouse: Indicates socio-economic factors.
hours-per-week: Higher work hours correlate with higher income.
education_level_per_hour: Combines education and work hours.
relationship_Husband: Highlights household structure's influence.
3. Visualizations
Boxplots:
age and hours-per-week distributions by income class.
Barplots:
Income distribution by gender, workclass, and race.