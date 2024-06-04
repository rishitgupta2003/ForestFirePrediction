# Forest Fire Prediction README

## Project Overview
This project aims to predict the occurrence of forest fires using machine learning models. The prediction system is built using Python and various machine learning libraries, and it is deployed as a web application using Flask.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Libraries Used](#libraries-used)
- [Contributing](#contributing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rishitgupta2003/forest-fire-prediction.git
   cd forest-fire-prediction
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask web application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`.

3. Upload a CSV file containing the data for prediction or input the data manually through the web interface.

## Dataset
The dataset used for training the model should contain features relevant to predicting forest fires. These features may include temperature, humidity, wind speed, and others. Ensure the dataset is in a CSV format and properly preprocessed.

## Model Training
Three machine learning models are used in this project:
- Decision Tree Classifier
- Random Forest Classifier
- Logistic Regression

The steps to train the models are as follows:
1. Load and preprocess the dataset using pandas and sklearn.
2. Split the dataset into training and testing sets.
3. Train the models on the training set.
4. Evaluate the models on the testing set using accuracy score.

The trained models are saved using pickle for later use in the web application.

## Web Application
The web application is built using Flask. It allows users to:
- Upload a CSV file with the data for prediction.
- View the prediction results.

### Flask Application Structure
- `app.py`: The main Flask application file.
- `templates/`: Directory containing HTML templates for the web pages.
  - `index.html`: The main page where users can upload the CSV file and view results.
- `static/`: Directory for static files (e.g., CSS, JS).

### Example Flask Routes
- `/`: The home route that renders the main page.
- `/create-model`:The route to generate model.
- `/predict-fire`: The route that handles the prediction logic.

## Libraries Used
- **Flask**: For building the web application.
- **os**: For interacting with the operating system.
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **sklearn**: For machine learning tasks.
  - `MinMaxScaler`: For feature scaling.
  - `train_test_split`: For splitting the dataset.
  - `DecisionTreeClassifier`, `RandomForestClassifier`, `LogisticRegression`: For building models.
  - `accuracy_score`: For evaluating model performance.
- **pickle**: For saving and loading trained models.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.