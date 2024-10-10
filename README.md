# Sleep Quality Prediction Project

This project is a **Streamlit** web application that predicts sleep quality based on several health and lifestyle factors, and provides visualizations of data from `Health_Sleep_Statistics.csv`.

## Features

1. **Sleep Quality Prediction**:
    - Users can input their age, gender, daily steps, calories burned, physical activity level, dietary habits, and sleep disorders.
    - The app uses a pre-trained model (`model.pkl`) to predict sleep quality as a percentage.
  
2. **Health Statistics Visualization**:
    - The app provides visualizations of relationships between different health statistics from the `Health_Sleep_Statistics.csv` file.
    - Current visualizations include:
      - **Age vs Sleep Quality** with a regression line
      - **Daily Steps vs Sleep Quality** with a regression line

## Installation and Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/sleep-quality-predictor.git
    ```

2. Navigate to the project directory:

    ```bash
    cd sleep-quality-predictor
    ```

3. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Ensure the following files are present in the project directory:
    - `model.pkl` (the pre-trained model for sleep quality prediction)
    - `Health_Sleep_Statistics.csv` (the dataset for generating visualizations)

## Running the Application

To run the Streamlit app locally:

```bash
streamlit run app.py
