# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Libraries:
- numpy
- pandas
- nltk
- sqlalchemy
- scikit_learn

### Motivation:
This project is part of Udacity Data Scientist Nano program. The original datasets are provided by FigureEight. 
The project is to analyze data and build a classifier model for disaster message.

### Acknowledgement:
I would like to thank FigureEight for providing the relevant data used in this project and Udacity for providint Data Scientist Nano Program.

