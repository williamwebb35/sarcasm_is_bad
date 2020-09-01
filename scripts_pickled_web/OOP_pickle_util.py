# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:29:11 2020

@author: sinua

This file contains the functions that pickle and unpickle files.

"""

import pickle

    # save the largest average sarcasm score
def save_largest_ave_score(largest_ave_score):
    from pipeline_capstone2 import Make_Pipeline

    write_file = open('largest_ave_score_pickled', 'wb')
    pickle.dump(largest_ave_score, write_file)

    # save the largest average sarcasm score's username
def save_largest_username(largest_username):
    from pipeline_capstone2 import Make_Pipeline

    write_file = open('largest_username_pickled', 'wb')
    pickle.dump(largest_username, write_file)

    # save the lowest average sarcasm score
def save_lowest_ave_score(lowest_ave_score):
    from pipeline_capstone2 import Make_Pipeline

    write_file = open('lowest_ave_score_pickled', 'wb')
    pickle.dump(lowest_ave_score, write_file)

    # save the lowest average sarcasm score's username
def save_lowest_username(lowest_username):
    from pipeline_capstone2 import Make_Pipeline

    write_file = open('lowest_username_pickled', 'wb')
    pickle.dump(lowest_username, write_file)

    #save the largest single tweet score
def save_largest_score(largest_score):
    from pipeline_capstone2 import Make_Pipeline

    write_file = open('largest_score_pickled', 'wb')
    pickle.dump(largest_score, write_file)

    #save the tweet with the largest single score
def save_largest_tweet(largest_tweet):
    from pipeline_capstone2 import Make_Pipeline

    write_file = open('largest_tweet_pickled', 'wb')
    pickle.dump(largest_tweet, write_file)

    #save the lowest single tweet score
def save_lowest_score(lowest_score):
    from pipeline_capstone2 import Make_Pipeline

    write_file = open('lowest_score_pickled', 'wb')
    pickle.dump(lowest_score, write_file)

    #save the tweet with the lowest single score
def save_lowest_tweet(lowest_tweet):
    from pipeline_capstone2 import Make_Pipeline

    write_file = open('lowest_tweet_pickled', 'wb')
    pickle.dump(lowest_tweet, write_file)

    # save a pipeline instance from Make_Pipeline method in pipeline_capstone2.py
def save_pipeline_object(pipeline_object):
    from pipeline_capstone2 import Make_Pipeline

    write_file = open('pipeline_object_pickled', 'wb')
    pickle.dump(pipeline_object, write_file)

    # save a preprocessor instance from Preprocessor method in Preprocessor.py
def save_preprocessor_instance(preprocessor_instance):
    from Preprocessor import Preprocessor

    write_file = open('preprocessor_instance_pickled', 'wb')
    pickle.dump(preprocessor_instance, write_file)

    # save a preprocessor instance of the sentiment140 data ....
        #....from Preprocessor method in Preprocessor.py
def save_sent_prepro_inst(preprocessor_instance):
    from Preprocessor import Preprocessor

    write_file = open('sent_prepro_inst_pickled', 'wb')
    pickle.dump(preprocessor_instance, write_file)

def save_clf(test):
	from OOP_clas_def import Predictor

	write_file = open('clf_bigrams_pickled', 'wb')
	pickle.dump(test, write_file)

    # pickles the fitted multinomial naive Bayes model fitted on the training..
        #... data in the 'train_and_fit' method of 'OOP_clas_def.py'
def save_nb_clf(test):
	from OOP_clas_def import Predictor

	write_file = open('nb_clf_bigrams_pickled', 'wb')
	pickle.dump(test, write_file)

def save_vectorizer(test):      # somehow pickle this from 'OOP_clas_def':  self._vectorizer = X
	from OOP_clas_def import Predictor

	write_file = open('vectorizer_bigrams_pickled', 'wb')
	pickle.dump(test, write_file)

def save_predictor(test):
	from OOP_clas_def import Predictor

	write_file = open('predictor_pickled', 'wb')
	pickle.dump(test, write_file)

    # pickles the df resulting from entering a single tweet to 'tweet_predictor' method of
    #... 'pipeline_capstone2.py'
def save_tweets_df(tweets_df):
	write_file = open('tweets_df_pickled', 'wb')
	pickle.dump(tweets_df, write_file)

    # pickles the df resulting from entering a username to 'tweet_predictor' method of
    #... 'pipeline_capstone2.py'
def save_usernames_df(usernames_df):

    write_file = open('usernames_df_pickled', 'wb')
    pickle.dump(usernames_df, write_file)

        # pickle list of average scores
def save_average_scores(average_scores):
    write_file = open('average_scores_pickled', 'wb')
    pickle.dump(average_scores, write_file)

    # pickle list of average scores
def save_username_score(username_score):
    write_file = open('username_score_pickled', 'wb')
    pickle.dump(username_score, write_file)
    
    # pickles a list created from combining all the words from all the ...
        #...preprocessed tweets
def save_tweet_words(tweet_words):
    write_file = open('tweet_words_pickled', 'wb')
    pickle.dump(tweet_words, write_file)

    # pickles DF resulting from of 'pred_word_probs' method from 'pipeline_capstone2.py'
def save_word_probs(word_probs_df):
    write_file = open('word_probs_pickled', 'wb')
    pickle.dump(word_probs_df, write_file)    


	    # load the largest average sarcasm score
def load_largest_ave_score(largest_ave_score):
    from pipeline_capstone2 import Make_Pipeline

    largest_ave_score = open(largest_ave_score, 'rb')
    return pickle.load(largest_ave_score)

    # load the username with the largest average sarcasm score
def load_largest_username(largest_username):
    from pipeline_capstone2 import Make_Pipeline

    largest_username = open(largest_username, 'rb')
    return pickle.load(largest_username)

    # load the largest single score
def load_largest_score(largest_score):
    from pipeline_capstone2 import Make_Pipeline

    largest_score = open(largest_score, 'rb')
    return pickle.load(largest_score)

    # load the tweet with the largest single score
def load_largest_tweet(largest_tweet):
    from pipeline_capstone2 import Make_Pipeline

    largest_tweet = open(largest_tweet, 'rb')
    return pickle.load(largest_tweet)

    # load the lowest average sarcasm score
def load_lowest_ave_score(lowest_ave_score):
    from pipeline_capstone2 import Make_Pipeline

    lowest_ave_score = open(lowest_ave_score, 'rb')
    return pickle.load(lowest_ave_score)

    # load the username with the lowest average sarcasm score
def load_lowest_username(lowest_username):
    from pipeline_capstone2 import Make_Pipeline

    lowest_username = open(lowest_username, 'rb')
    return pickle.load(lowest_username)

    # load the lowest single score
def load_lowest_score(lowest_score):
    from pipeline_capstone2 import Make_Pipeline

    lowest_score = open(lowest_score, 'rb')
    return pickle.load(lowest_score)

    # load the tweet with the lowest single score
def load_lowest_tweet(lowest_tweet):
    from pipeline_capstone2 import Make_Pipeline

    lowest_tweet = open(lowest_tweet, 'rb')
    return pickle.load(lowest_tweet)

    # load pipeline object
def load_pipeline_object(file_path):
    from pipeline_capstone2 import Make_Pipeline

    pipeline_object = open(file_path, 'rb')
    return pickle.load(pipeline_object)

    # load preprocessor instance including attributes such as a df from Preprocessor method in Preprocessor.py
def load_preprocessor_instance(file_path):
    from Preprocessor import Preprocessor

    preprocessor_instance = open(file_path, 'rb')
    return pickle.load(preprocessor_instance)

def load_clf(file_path):
	#from OOP_clas_def import Predictor

	doc_file = open(file_path, 'rb')
	return pickle.load(doc_file)

    # created for loading the pickled multinomial naive Bayes model fitted on the...
        #... training data in the 'train_and_fit' method of 'OOP_pickle_util.py
def load_nb_clf(file_path):
	#from OOP_clas_def import Predictor

	doc_file = open(file_path, 'rb')
	return pickle.load(doc_file)

def load_vectorizer(file_path):
	#from OOP_clas_def import Predictor

	doc_file = open(file_path, 'rb')
	return pickle.load(doc_file)

def load_predictor(file_path):
	#from OOP_clas_def import Predictor

	doc_file = open(file_path, 'rb')
	return pickle.load(doc_file)
     
def load_tweets_df(file_path):

    doc_file = open(file_path, 'rb')
    return pickle.load(doc_file)

def load_usernames_df(file_path):

    doc_file = open(file_path, 'rb')
    return pickle.load(doc_file)

    # load pickled list of average scores
def load_average_scores(file_path):
    doc_file = open(file_path, 'rb')
    return pickle.load(doc_file)

    # load pickled username score
def load_username_score(file_path):
    doc_file = open(file_path, 'rb')
    return pickle.load(doc_file)

def load_tweet_words(file_path):
    doc_file = open(file_path, 'rb')
    return pickle.load(doc_file)

# unpickles DF resulting from of 'pred_word_probs' method from 'pipeline_capstone2.py'
def load_word_probs(file_path):
    doc_file = open(file_path, 'rb')
    return pickle.load(doc_file)    





