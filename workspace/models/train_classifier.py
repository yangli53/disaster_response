import sys
import nltk
nltk.download(['stopwords', 'punkt', 'wordnet'])
import re
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Load data from SQLite and return X, Y and category_names.
    
    Input: database_filepath - path of the database
    
    Output: X, Y, category_names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath, con=engine)
    X = df['message']
    Y = df.drop(['message', 'original', 'genre', 'id'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Normalize, tokenize, clean and lemmatize the text.
    
    Input: text - raw text
    
    Output: lemmed_tokens - lemmed clean tokens
    """
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    tokens = word_tokenize(text)
    clean_tokens = [token for token in tokens if token not in stopwords.words('english')]
    
    lemmatizer = WordNetLemmatizer()
    lemmed_tokens = [lemmatizer.lemmatize(token) for token in clean_tokens]
    
    return lemmed_tokens


def build_model():
    """
    Use pipeline and gridsearch to build a model.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [5, 10]
    }
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Return the f1 score, precision and recall for each output category.
    
    Input: model - model built from last step
           X_test
           Y_test
           category_names
   
   Output: report of f1 score, precision and recall for each category
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(classification_report(Y_test[col], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save the model as a pickle file.
    
    Input: model
           model_filepath - file path to save the model          
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