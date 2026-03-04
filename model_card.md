# Model Card: Census Income Classifier

## Model Details
- **Developed by:** Sary Rodriguez
- **Model type:** Random Forest Classifier
- **Language:** Python 3.8
- **Libraries:** scikit-learn, pandas, numpy

## Intended Use
- **Primary use:** Predict whether a person earns more or less than $50K/year
- **Intended users:** Data scientists, ML engineers
- **Out-of-scope:** Should not be used for making real hiring or financial decisions

## Training Data
- **Dataset:** UCI Adult Census Income Dataset
- **Size:** ~32,000 rows
- **Features:** age, workclass, education, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country
- **Target:** salary (<=50K or >50K)
- **Preprocessing:** Removed leading/trailing spaces, label encoded categorical features

## Evaluation Data
- 20% of the dataset was held out for evaluation using train_test_split with random_state=42

## Metrics
- **Precision:** ~0.85
- **Recall:** ~0.70
- **F1 Score:** ~0.77

## Ethical Considerations
- The dataset contains sensitive attributes like race, sex, and native-country
- The model may reflect historical biases present in the census data
- Results should not be used to discriminate against individuals

## Caveats and Recommendations
- Performance may vary across demographic slices (see slice_output.txt)
- Model should be retrained periodically with more recent census data
- Not recommended for use as a sole decision-making tool
