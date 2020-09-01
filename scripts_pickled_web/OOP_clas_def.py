# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:29:11 2020

@author: sinua

This file contains the functions that train and fit the model.


Reference: https://towardsdatascience.com/deploying-models-to-flask-fb62155ca2c4
Reference: https://github.com/mmalinas/Springboard_Git/blob/master/Capstone2_MelanieM/website-final/class_def.py
"""

""" IMPORTS """


import pandas as pd
import re # for using regular expressions to remove numbers
#import string # for removing punctuation
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import OOP_pickle_util
import en_core_web_sm
nlp = en_core_web_sm.load()
import dill
from Preprocessor import Preprocessor

""" VARIABLES DEFINED OUTSIDE THE CLASS"""

lemmatizer=WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

    # List of US state abbreviations
state_abbrevs = ["AL", "Al", "AK","Ak", "AZ","Az", "AR","Ar", "CA","Ca", "CO","Co", "CT","Ct", "DC","Dc", "DE", "De","FL","Fl", "GA", "Ga",
          "HI","Hi", "ID", "Id","IL","Il", "IN", "In","IA","Ia", "KS","Ks", "KY","Ky", "LA","La", "ME", "Me","MD", "Md",
          "MA","Ma", "MI", "Mi","MN","Mn", "MS", "Ms","MO","Mo", "MT","Mt", "NE","Ne", "NV", "Nv","NH","Nh", "NJ", "Nj",
          "NM","Nm", "NY","Ny", "NC","Nc", "ND","Nd", "OH", "Oh","OK","Ok", "OR","Or", "PA","Pa", "RI","Ri", "SC", "Sc",
          "SD", "Sd","TN","Tn", "TX","Tx", "UT","Ut", "VT","Vt", "VA","Va", "WA","Wa", "WV","Wv", "WI","Wi", "WY", "Wy"]

    # List of US State names
state_names = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado",
  "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois",
  "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
  "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
  "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",
  "North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
  "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
  "Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]

    # List of abbreviations for the United States
US_names = ["US","U.S.","United States","United States of America", "America"]

    # location data used to filter tweets
US_locs = state_abbrevs + state_names + US_names  

    # specific words to remove from tweets (Added 'Sarcastic' and 'sarcastic' during...
        #... analysis of second round of tweets collected
bad_words = [ 'Sarcastic', 'sarcastic', 'Sarcasm','sarcasm', 'nsarcasm', 'nnsarcasm', '_', '__', '___', '____','______','___________________', '______________________']

"""Initialize the Predictor class with csv files containing both sarcastic tweets and
    non-sarcastic tweets from the same usernames """
       
class Predictor:
    def __init__(self):
        self = self
            
    """ Trains and fits the model. Note: requires Preprocessor.tweet_format() to be run 
        first on the training data ('all_sarc_and_matching_tweets.csv') as occurs in the 'main' function of 'Preprocessor.py'
        so the training df is already available as an attribute in the 'train_and_fit' method """
        
    def train_and_fit(self):
        print("Executing the 'train_and_fit' method which trains and tests the model.")
            # saves the vectorizer instance as an attribute; use 'min_df = 3' argument or similar as needed
        self.vectorizer = CountVectorizer()
            # load the preprocessor instance including attributes, most importantly the preprocessed DF...
                #... which along with the other attributes was generated from the...
                    #... training data ('all_sarc_and_matching_tweets.csv') as the 'tweet_data'...
                        #... argument for class Preprocessor in 'Preprocessor.py'
        preprop_inst = OOP_pickle_util.load_preprocessor_instance('preprocessor_instance_pickled') 
            # assign the preprocessed DF an attribute of the current class Predictor instance
        self.df = preprop_inst.df
        
        print("Checking the DF header:",self.df.head())
            # create the feature array for the model using the third to last column which ...
                 #... contains the final processed tweet text   
        self.X = self.df.iloc[:,-3]
             # create the target array which identifies whether tweets are sarcastic or not
        self.y = (self.df['tweet_cat']).values.astype(np.int)      
            # create training and test splits. The following represents the split-first approach ...
            # where the vectorizer and classifier are fit to the training data and the testing data...
            # are essentially novel data evaluated by the model based on the training data. The feature...
            #... data at this stage (X-train and X-test) are still in text form
        X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
		#best_alpha = self.get_best_alpha()
            # learn the vocabulary and create a numerical document-term matrix (DTM) ...
                #...for the training split features based on the text data in X_train
        self.X_train = self.vectorizer.fit_transform(X_train)
            # create a document-term matrix (DTM) based on the training data ....
                #... using the text data in X_test. This intentionally neglects....
                # ... any new vocabulary in the testing data (X_test) not already ...
                #... found in the training text data (X_train)
        self.X_test = self.vectorizer.transform(X_test)
            # logistic regression with C = 1 performed the best compared to other algorithms as discovered in 'Gridsearch_Capstone2.ipnyb'
            # fit the logistic regression model to the training split features and training split target arrays
            # set 'max_iter = 1000' up from the default 100 to avoid the 'ConvergenceWarning' error
        self.clf = LogisticRegression(C = 1, max_iter=1000).fit(self.X_train, self.y_train) # saves the clf as an attribute of the current instance
            # Save a pickled version of the trained logistic regression model from...
                #...above. The pickled logistic regression model is then used as...
                    #... one of the steps in the logistic regression pipeline to ...
                        #... evaluate a novel username in 'pipeline_capstone2.py'
        OOP_pickle_util.save_clf(self.clf) 
            # multinomial naive Bayes added as attribute of the current instance
                # added in order to predict the sarcasm probability...
                    #...of individual words in the most sarcastic tweets for a specific...
                        #... username and populate the third synced carousel in the web app
        self.nb_clf= MultinomialNB().fit(self.X_train, self.y_train)
            # Save a pickled version of the trained multinomial naive Bayes model from...
                #...above. The pickled multinomial naive Bayse model is then used as...
                    #... one of the steps in the multinomial naive Bayes pipeline to ...
                        #... evaluate a novel username in 'pipeline_capstone2.py'
        OOP_pickle_util.save_nb_clf(self.nb_clf) 
            # pickles the vectorizer which is used as one of the steps in both the...
                # logistic regression and multinomial naive Bayes pipelines in...
                    # 'pipeline_capstone2.py'
        OOP_pickle_util.save_vectorizer(self.vectorizer) #save vectorizer 
        
        return self.X, self.y, self.clf, self.nb_clf, self.vectorizer, self.df 
    
    """  conducts 5-fold cross-validation, gridsearching and prints results of 
    these tests for the model"""
    
    def scores(self):
            # call the train_and_fit method to make those attributes available
        self.train_and_fit()
            # Print the accuracy of the logistic regression model on the training dataset
        print("Classification score:",self.clf.score(self.X_test, self.y_test))
        conf_mat = (confusion_matrix(self.y_test, self.clf.predict(self.X_test)))
        print(conf_mat)      
            # instantiate CountVectorizer 
        vectorizer = CountVectorizer()
            # vectorize the full X column, creating a document-term matrix needed for cross-validation
        X_dtm = vectorizer.fit_transform(self.X)
            # generate cross-validation scores to simulate model performance on unseen data
        self.cv_scores = cross_val_score(self.clf, X_dtm, self.y, cv=5)
        print("Cross validation scores:", self.cv_scores)
        print("Mean cross validation score:", self.cv_scores.mean())       
            # generate a classification report
        tweet_type_predictions = self.clf.predict(self.X_test)
        self.clas_report = classification_report(self.y_test, tweet_type_predictions, digits=4)
        print(classification_report(self.y_test, tweet_type_predictions, digits=4))
        return self.cv_scores, self.clas_report
    
   
        

if __name__ == '__main__':
    from OOP_clas_def import Predictor
    tweet_predictor = Predictor()
    tweet_predictor.train_and_fit()
    # scores = collected_tweets.cv_score() # method needs updating
    #clf = tweet_predictor.clf
    #nb_clf = tweet_predictor.nb_clf
    #OOP_pickle_util.save_clf(clf) # save logistic regression model created during the 'def train_and_fit' ...
    #... method 
    #OOP_pickle_util.save_nb_clf(nb_clf) # save multinomial naive Bayes model created during the 'def train_and_fit' ...
    #... method 
    #vectorizer = tweet_predictor.vectorizer
    #OOP_pickle_util.save_vectorizer(vectorizer) #save vectorizer 
    print('Predictor success!')
    #tweet_predictor.scores()


""" DRAFT / CUT ITEMS BELOW:
    
    "Use grid-searching to find the best hyperparameter alpha for multinomial naive Bayes"
    
    def get_alpha(self):
           
            # GridSearchCV
        parameters = {'alpha':[0.01, 0.05, 0.1, 0.15, 0.5, 1, 1.5, 2.5, 5, 7.5, 10, 15, 50]}
        MNB = MultinomialNB()
        vectorizer = CountVectorizer()
        X_dtm = vectorizer.fit_transform(self.X)
        clf_grid = GridSearchCV(MNB, parameters)
        search = clf_grid.fit(X_dtm, self.y)
        best_params = search.best_estimator_
        self.best_alpha = best_params.alpha
        self.grid_results = search.cv_results_
        print(clf_grid.cv_results_)
        return self.best_alpha, self.grid_results
    
        """