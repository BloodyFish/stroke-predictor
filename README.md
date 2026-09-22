# Stroke Predictor

A small Python machine-learning project that explores how health and lifestyle factors can be used to estimate stroke risk. The model is trained with a **Random Forest classifier** using the Kaggle Stroke Prediction Dataset.

> [!WARNING]
> **This project is for educational purposes only.** Its predictions are not medical advice, a diagnosis, or a substitute for a qualified healthcare professional. Do not use it to make healthcare decisions.

## 🤔 What it does

The program:

1. Loads the stroke prediction dataset.
2. Converts categorical values into numeric features.
3. Fills missing BMI values with the dataset median.
4. Trains a Random Forest classifier.
5. Evaluates the model on a held-out portion of the dataset.
6. Prompts for an individual's information and prints a predicted result.

The model uses the following features:

- Gender
- Age
- Hypertension
- Heart disease
- Marital status
- Work type
- Residence type
- Average glucose level
- BMI
- Smoking status

## 📊 Model performance

Using the current training configuration and dataset split, the model achieved:

```text
Model Accuracy: 94.72%
```

> Accuracy can vary depending on the dataset, preprocessing steps, and training configuration. Because the dataset is imbalanced, accuracy alone should not be used to judge the model's overall performance.

## 📋 Requirements

- Python 3.9 or newer
- pandas
- scikit-learn

Install the dependencies with:

```bash
python -m pip install pandas scikit-learn
```

## 📋 Dataset setup

This project uses the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle.

Because the dataset has its own licensing terms, it is **not included in this repository**. Download it from Kaggle and place the CSV at the following path:

```text
data/healthcare-dataset-stroke-data.csv
```

Your project should look like this:

```text
stroke-predictor/
├── data/
│   └── healthcare-dataset-stroke-data.csv
├── main.py
└── README.md
```

## 💻 Run the project

From the repository root, run:

```bash
python main.py
```

The script will first train and evaluate the model, then ask you to enter the features required for a prediction. Use the numeric options shown in each prompt—for example, `0` or `1` for yes/no fields.

## 📋 Important limitations

This is a learning project and should not be treated as a production-ready clinical model. In particular:

- The dataset is imbalanced, so accuracy alone does not fully describe model performance.
- Results may vary depending on the dataset and training configuration.
- The model has not been clinically validated.
- A prediction of “No” does not mean that someone cannot have a stroke.

## 🤔 Generative AI Disclaimer
A lot of this code was written with the help of Gemini. This project was created as a means to teach myself machine learning using python, so I used Gemini to help me learn by doing.

## License

The source code is available under the [MIT License](LICENSE). The dataset is provided separately by Kaggle and remains subject to its own license and terms of use.
