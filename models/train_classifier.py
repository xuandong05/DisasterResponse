import sys
import nltk
nltk.download(['punkt','wordnet', 'averaged_perceptron_tagger'])

import numpy as np
import pandas as pd
import re
import sys
import os
import pickle
import sqlalchemy
from sqlalchemy import create_engine
import sqlite3
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn. multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    Load data from database and return the data
    Parameters:
    database_filepath(string): Filepth to SQlite database file
    Returns
    X: A dataframe of Independent Variables
    Y: A dataframe of Dependent Variable
    category_names:List of categories names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df', engine)
    X = df['message'].values
    Y = df.iloc[:, 4:].values
    category_names = (df.columns[4:]).tolist()
    return X, Y, category_names

    


def tokenize(text):
    """
    Tokenize message data
    Parameters:
    text: string
    
    Returns:
    clean_tokens: List of tokens
    """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Build pipeline and Classisfier
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__min_samples_leaf': [2, 5, 10],
        'clf__estimator__max_depth': [10, 50, None]
    }
    
    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs = -1, verbose = 10)
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate models performace
    
    Parameters:
    model (Classification): stored classification model
        X_test (string): Independent Variables
        Y_test (string): Dependent Variables
        category_names (DataFrame): List of category strings
    Returns:
    None
    """
    Y_pred = model.predict(X_test)
    print("Classification Report per Category:\n")
    for i in range(len(category_names)):
        print("Category:", category_names[i])
        print(classification_report(Y_test[:, i], Y_pred[:, i]))



def save_model(model, model_filepath):
    """
    Save classification model to pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()