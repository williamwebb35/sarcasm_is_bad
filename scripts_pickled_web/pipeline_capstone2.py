# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:02:50 2020

@author: sinua

Runs a pipeline for analyses related to capstone 2
Evaluates scores and average scores for single tweets and usernames, respectively...
... and determines if the score or average represents a new record low or ...
.. new record high score

"""
from sklearn.pipeline import Pipeline as Pipeline
from Preprocessor import Preprocessor
from OOP_clas_def import Predictor
#from OOP_predictor_api import API
import OOP_pickle_util
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.feature_extraction.text import CountVectorizer
from decimal import *
from OOP_pickle_util import save_word_probs
from OOP_pickle_util import load_tweet_words
import pandas as pd

# guppy provides an estimate of memory use
#import guppy
#from guppy import hpy
#h = hpy()
#print("Guppy and Heapy from 'pipeline_capstone2.py",h.heap())

# Pipeline object 
class Make_Pipeline:
    def __init__(self):
        self = self
    
            # pickled vectorizer, LR classifier and MNB classifier 
            # load the pickled vectorizer fit and transformed on the training data ...
            #... set as an attribute
        self.vectorizer = OOP_pickle_util.load_vectorizer('vectorizer_bigrams_pickled') 
            # load the pickled classifier trained on the training data training set....
            #... as an attribute
        self.classifier = OOP_pickle_util.load_clf('clf_bigrams_pickled')
            # loads the pickled multinomial naive Bayes classifier fit on the ...
                #... training data in the 'train_and_fit' methed of 'OOP_clas_def.py'
        self.nb_classifier = OOP_pickle_util.load_clf('nb_clf_bigrams_pickled')
    
                
            # load the largest average tweet score as an attribute
        self.largest_ave_score =  OOP_pickle_util.load_largest_ave_score('largest_ave_score_pickled')        
            # load the username with the largest average tweet score as an attribute
        self.largest_username =  OOP_pickle_util.load_largest_username('largest_username_pickled') 
            # load the lowest average tweet score as an attribute
        self.lowest_ave_score =  OOP_pickle_util.load_lowest_ave_score('lowest_ave_score_pickled') 
            # load the username with the lowest average tweet score as an attribute
        self.lowest_username =  OOP_pickle_util.load_lowest_username('lowest_username_pickled')     
            # load the largest single tweet score as an attribute
        self.largest_score =  OOP_pickle_util.load_largest_score('largest_score_pickled')  
            # load the tweet with the largest single tweet score as an attribute
        self.largest_tweet =  OOP_pickle_util.load_largest_tweet('largest_tweet_pickled') 
            # load the lowest single tweet score as an attribute 
        self.lowest_score =  OOP_pickle_util.load_lowest_score('lowest_score_pickled') 
            # load the tweet with the lowest single tweet score as an attribute
        self.lowest_tweet =  OOP_pickle_util.load_lowest_tweet('lowest_tweet_pickled')
            # load pickled list of average scores as an attribute
        self.average_scores = OOP_pickle_util.load_average_scores('average_scores_pickled')
            # load existing df created from single tweets as an attribute
        self.tweets_df = OOP_pickle_util.load_tweets_df('tweets_df_pickled')

    
    def predict_tweet(self,tweet_data, username): # added username so the username can be included in self.df and ...
            #...the df that collects the most sarcastic tweets
        
            # handling an empty string
        if tweet_data == '':
                # Instantiate the preprocessor with a novel tweet 
            processed_tweet = Preprocessor(tweet_data) 
                # Call the 'tweet_format' method on the preprocessor instance and preprocess the novel tweet
            processed_tweet.tweet_format()
                # Assign the preprocessed tweets DF as a pipeline object attribute
            self.df = processed_tweet.df
            
            print("EMPTY STRING DETECTED")
            prediction = 0
            ave_sarc_score = 0
            self.ave_sarc_score = 0
            
            print("Average sarcasm score:", self.ave_sarc_score) 
            prob_sarc_tweet = 0
            print("The tweet entered is empty")
            prob_d = {'prob_sarc':prob_sarc_tweet}
            print('prob_d:',prob_d)    
            
            self.df = self.df.sort_values('prob_sarc2', axis = "columns", ascending = False)
                # reset index to allow for accurate indexing of the most sarcastic score, etc.
            self.df = self.df.reset_index(drop=True)
            
            return(tweet_data, prob_sarc_tweet, self.df, ave_sarc_score)
            
            #x_input, predictions, df, ave_sarc_score = pipe_object(y) 
        
        else:
                # Instantiate the preprocessor with a novel tweet 
            processed_tweet = Preprocessor(tweet_data) 
            #processed_tweet = Preprocessor(self.tweet_data)
                # Call the 'tweet_format' method on the preprocessor instance and preprocess the novel tweet
            processed_tweet.tweet_format()
            
                # Assign the sequence of analyses for the logistic regression pipeline
            steps = [('vect', self.vectorizer), ('clf', self.classifier)]
                # Assign the sequence of analyses for the multinomial naive Bayes pipeline
            steps_nb = [('vect', self.vectorizer), ('clf_nb', self.nb_classifier)]
                # Instantiate a pipeline object for the logistic regression analysis
            pipe = Pipeline(steps)
                # Instantiate a pipeline object for the multinomial naive Bayes analysis
            pipe_nb = Pipeline(steps_nb)
            
                # Assign a pipeline instance and predict the probability of the novel tweet being sarcastic
                # 'prediction' returns a value only for the first row of 'processed_tweet.df'
            prediction = pipe.predict_proba(processed_tweet.df.text_final)
                
                # Assign the preprocessed tweets DF as a pipeline object attribute...
                    #.. to be used for the logistic regression analysis
            self.df = processed_tweet.df
                        
                # Produce a prediction for each row of the 'text' column in the DF ...
                # ... which contains a single row for individual tweets but multiple...
                # ... rows when usernames are entered as 'tweet_data'
                
            self.df['text_final'] = self.df['text_final'].apply(lambda x: [x]) # Prevents the ..
                #...following error when calling pipeline on each row with a lmbda function:....
                #..."#ValueError: Iterable over raw text documents expected, string object received."
                
                # Produce a prediction for each row of the 'text' column in 'processed_tweet.df'
            self.df['prob_sarc'] = self.df['text_final'].apply(lambda x: pipe.predict_proba(x))  
            
                # create a new column for the probability which just has the score for sarcasm 
            self.df['prob_sarc2'] = self.df['prob_sarc'].apply(lambda x: x[0][1])
            
                # calculate mean sarcasm score based on 'prob_sarc2' column
                # append this to the list of average scores when usernames evaluated
            self.mean_sarc_score = (round((self.df.prob_sarc2.mean()),2)*100)       
             
                # calculate median sarcasm score based on 'prob_sarc2' column
            self.median_sarc_score = (round((self.df.prob_sarc2.median()),2)*100)
                
                # sort the df by the scores  
            self.df = self.df.sort_values('prob_sarc2', ascending = False)
            
                # reset index to allow for accurate indexing of the most sarcastic score, etc.
            self.df = self.df.reset_index(drop=True)
            
                # Assign the first five rows of the sorted preprocessed tweets DF ...
                    #...as a pipeline object attribute to be used for the multinomial ...
                        #...naive Bayes analysis
            self.df_nb = self.df.iloc[0:5]
            
                # Produce a prediction for each row of the 'text' column in 'processed_tweet.df_nb'...
                    #... using multinomial naive Bayes
            self.df_nb['prob_sarc_nb'] = self.df_nb['text_final'].apply(lambda x: pipe_nb.predict_proba(x))  
            
                # create a new column for the probability which just has the score for sarcasm ...
                    #... created using multinomial naive Bayes
            self.df_nb['prob_sarc2_nb'] = self.df_nb['prob_sarc_nb'].apply(lambda x: x[0][1])
            
                # convert 'text' column with a single string in the list to a column consisting of list of words
            self.df_nb['text_words'] = self.df_nb['text_final'].apply(lambda x:[sentence.split() for sentence in x])
            
                # removes the nested list structure 
            self.df_nb['text_words'] = self.df_nb['text_words'].apply(lambda x: x[0])
            
                # Execute the multinomial naive Bayes pipeline on the 'text_words' column        
            self.df_nb['test_nb2'] = [pipe_nb.predict_proba(i) for i in self.df_nb['text_words']]
            
                # Create a new column with just the sarcasm score based on the ...
                    #... multinomial naive Bayes pipeline analysis
            
            self.df_nb['mnb_sarc'] = self.df_nb['test_nb2'].apply(lambda x:[item[1] for item in x] )
            
                        
                # identify the most sarcastic tweet in the df            
            self.most_sarcastic_tweet = self.df.text_final[0]
            
            print("Most sarcastic tweet:",self.most_sarcastic_tweet)
            
                # identify the score for the most sarcastic tweet in the df
            self.most_sarcastic_score = self.df.prob_sarc2[0]
            
            print("The score for the most sarcastic tweet from this username is: ",self.most_sarcastic_score)
             
                # add the username to each row of self.df 
            self.df['username'] = username
            
                # Calculate mean sarcasm score for 'prob_sarc" column of self.df 
            self.ave_sarc_score = (round((self.df.prob_sarc.mean()[0][1]),2))*100
           
            ave_sarc_score = self.ave_sarc_score
            
                # add the average score as a column
            self.df['ave_sarc_score'] = self.df.prob_sarc2.mean()
            
            print("Average sarcasm score:", self.ave_sarc_score)
            
           # print("Median sarcasm score:", self.med_sarc_score)
            
            prob_sarc_tweet = round(100*prediction[0][1])
            print("Here is the original tweet:", tweet_data)
            prob_d = {'prob_sarc':prob_sarc_tweet}
            print('prob_d:',prob_d)    
            print("There is a",float(Decimal(prob_sarc_tweet).quantize(Decimal('.01'), rounding=ROUND_DOWN)),"% probability the tweet is sarcastic.")
            print('New DF', self.df)
            
            # capture the most sarcastic tweet and add it to the df that collects...
            #... the most sarcastic tweets
            # steps:
            # unpickle the df that collects the most sarcastic tweet
            # add the first row of self.df to the df that collects the most sarcastic tweets
            # pickle the df that collects the most sarcastic tweets
            
            # Conditional statements that evaluate whether 'tweet_data' is a string...
            #... and if so, determines whether it is a single tweet or a username. If...
            #... a single tweet, then determines if the score represents a new high...
            #... or low record. If a username, determines if the average represents...
            #.. a new high or low record. If 'tweet_data' is not a string, then it...
            #... is a list or .csv which would only be input during development...
            #... and checking for record scores is not necessary
                   
            if type(tweet_data) == str:
                print("string detected")
                
                    # Evaluate whether the average score represents a new high or low...
                    #... record
                    # unpickle the list of average scores for usernames, append the current...
                    #... score and pickle the list again
                if tweet_data.startswith('@'):
                    print("username detected")
                    """
                    # draft code for pickling the username data so the top and lowest...
                    #... usernames can be displayed
                    
                    # unpickle the existing df created from usernames
                    usernames_df = OOP_pickle_util.load_usernames_df('usernames_df_pickled')
                     
                    # select the first row for the username data since the df is sorted, this will be the ...
                    #... most sarcastic tweet
                    new_usernames_df = self.df[:1]
                    
                    # create new column of rounded values of the 'ave_sarc_score' column
                    new_usernames_df['score_rounded'] = round(new_usernames_df['ave_sarc_score'],2)
                    
                    # drop the 'prob_sarc' and 'text' columns because they are lists and will prevent...
                    #... the drop duplicates function from working. Drop the 'ave_sarc_score' column because...
                    #... of too much variation within usernames cause slight variation and more than one...
                    #... entry per username
                    new_usernames_df = new_usernames_df.drop(['prob_sarc','text_final','ave_sarc_score'], axis = 1)
                    
                    # concat the existing df with the first row of self.df which contains the ...
                    #... most sarcastic tweet by the username
                    usernames_df = pd.concat([usernames_df, new_usernames_df])
                    
                    
                    # drop duplicatesF
                    usernames_df = usernames_df.drop_duplicates()
                    
                    # sort the combined df by the sarcasm score, 'score_rounded'
                    usernames_df = usernames_df.sort_values(by=['score_rounded'], ascending =False)
                    
                    # pickle the resulting df
                    OOP_pickle_util.save_usernames_df(usernames_df)
                    """
                    
                        # load pickled list of average scores
                    self.average_scores = OOP_pickle_util.load_average_scores('average_scores_pickled')
                        # append the current username's average score to the existing...
                        #... list of average username scores                    
                    self.average_scores.append(self.mean_sarc_score)
                        # pickle the list of average scores
                    OOP_pickle_util.save_average_scores(self.average_scores)
                    
        
                        # determine if the average score is a new record high
                    if  ave_sarc_score > self.largest_ave_score:
                        self.largest_ave_score = ave_sarc_score   
                        print("NEW LARGEST AVERAGE SCORE!!!:", self.largest_ave_score)          
                        #... and pickle new largest ave score
                        OOP_pickle_util.save_largest_ave_score(self.largest_ave_score) 
                        # pickle username with the largest average score
                        self.largest_username = tweet_data 
                        OOP_pickle_util.save_largest_username(self.largest_username)
                        print("NEW LARGEST SCORING USERNAME!!!:", tweet_data)
                    
                        # determine if the average score is a new record low
                         
                    if  ave_sarc_score < self.lowest_ave_score:
                        
                        self.lowest_ave_score = ave_sarc_score 
                        print("NEW LOWEST AVERAGE SCORE!!!:", self.lowest_ave_score)          
                        #... and pickle the new lowest average score
                        OOP_pickle_util.save_lowest_ave_score(self.lowest_ave_score)
                        # pickle username with lowest average score
                        OOP_pickle_util.save_lowest_username(tweet_data)
                        print("NEW LOWEST SCORING USERNAME!!!:", tweet_data)   
                
                    # Determine if a single tweet represents a new record high or record...
                    #... low score
                else:
                    print("single tweet detected")
                    
                    # unpickle the existing df created from single tweets
                    #tweets_df = OOP_pickle_util.load_tweets_df('tweets_df_pickled')
                    
                    # name the new df created from a single tweet submitted
                    new_tweets_df = self.df
                    
                        # create new column of rounded values of the 'ave_sarc_score' column
                    new_tweets_df['score_rounded'] = round(new_tweets_df['ave_sarc_score'],2)
                    
                        # drop the 'prob_sarc' and 'text' columns because they are lists and will prevent...
                        #... the drop duplicates function from working. Drop the 'ave_sarc_score' column because...
                        #... of too much variation within usernames cause slight variation and more than one...
                        #... entry per username
                    new_tweets_df = new_tweets_df.drop(['prob_sarc','text_final','ave_sarc_score'], axis = 1)
                    
                        # concat the exisiting df with self.df
                    self.tweets_df = pd.concat([self.tweets_df, new_tweets_df])
                    
                        # drop duplicates
                    self.tweets_df = self.tweets_df.drop_duplicates()
                    
                        # sort the combined df by the sarcasm score, 'score_rounded'
                    self.tweets_df = self.tweets_df.sort_values(by=['score_rounded'], ascending =False)
                                    
                        # pickle the resulting df created from single tweets
                    OOP_pickle_util.save_tweets_df(self. tweets_df)# this should be the df after concatenating ...
                        # ... self.df with the upickled df)

                        # Compare the current score with the record high score
                    if prob_sarc_tweet > self.largest_score:
                    #if prediction[0][1] > largest_score:
                        self.largest_score = prob_sarc_tweet 
                        print("NEW LARGEST SCORE!!!:", self.largest_score)          
                            #... and pickle prediction as the largest score
                        OOP_pickle_util.save_largest_score(self.largest_score) 
                            # pickle tweet as the largest tweet
                        OOP_pickle_util.save_largest_tweet(tweet_data)
                        print("NEW LARGEST SCORING TWEET!!!:", tweet_data)
                    
                        # Compare current score with the record low score
                    if prob_sarc_tweet < self.lowest_score:
                    #if prediction[0][1] < lowest_score:
                        print('previous lowest score:', self.lowest_score)
                        self.lowest_score = prob_sarc_tweet 
                        print("NEW LOWEST SCORE!!!:", self.lowest_score)          
                            #... and pickle prediction as the lowest score
                        OOP_pickle_util.save_lowest_score(self.lowest_score)
                        # pickle tweet as the largest tweet
                        OOP_pickle_util.save_lowest_tweet(tweet_data)
                        print("NEW LOWEST SCORING TWEET!!!:", tweet_data)
                        
                  # tweet data is the original username or tweet entered  
                  # prob_sarc_tweet is the probability the first row of the ...
                #... text column in self.df is sarcastic
                #... ave_sarc_score is the average sarcastic probability...
                #... score for the text column of self.df
            
            self.df = self.df.sort_values('prob_sarc2', ascending = False)
            
            return(tweet_data, prob_sarc_tweet, self.df, self.df_nb, ave_sarc_score,
                   self.mean_sarc_score, self.median_sarc_score, 
                   self.most_sarcastic_tweet, self.most_sarcastic_score)  

        # special method designed to assign sarcasm scores to indivdual words...
    def pred_word_probs(self, words_lst):#, df):
        
            # Assign the sequence of analyses for the logistic regression pipeline
        steps = [('vect', self.vectorizer), ('clf', self.classifier)]
                # Assign the sequence of analyses for the multinomial naive Bayes pipeline
        steps_nb = [('vect', self.vectorizer), ('clf_nb', self.nb_classifier)]
                # Instantiate a pipeline object for the logistic regression analysis
        pipe = Pipeline(steps)
                # Instantiate a pipeline object for the multinomial naive Bayes analysis
        pipe_nb = Pipeline(steps_nb)
            
                # Assign a pipeline instance and predict the probability of the novel tweet being sarcastic
                # 'prediction' returns a value only for the first row of 'processed_tweet.df'
        #prediction = pipe.predict_proba(processed_tweet.df.text)
        
        #words_lst = load_tweet_words('tweet_words_pickled')
        self.df = pd.DataFrame(words_lst)
        self.df = self.df.rename(columns={0:'text_final'})
                # Assign the df which is a method argument as a pipeline object attribute...
                    #.. to be used for the logistic regression analysis
        #self.df = df
                        
                # Produce a prediction for each row of the 'text' column in the DF ...
                # ... which contains a single row for individual tweets but multiple...
                # ... rows when usernames are entered as 'tweet_data'
               
        self.df['text_final'] = self.df['text_final'].apply(lambda x: [x]) # Prevents the following error when calling pipeline on each row with a lmbda function: "#ValueError: Iterable over raw text documents expected, string object received."
                # Produce a prediction for each row of the 'text' column in 'processed_tweet.df'
        self.df['prob_sarc'] = self.df['text_final'].apply(lambda x: pipe.predict_proba(x))  
            
                # create a new column for the probability which just has the score for sarcasm 
        self.df['prob_sarc2'] = self.df['prob_sarc'].apply(lambda x: x[0][1])
            
                # calculate mean sarcasm score based on 'prob_sarc2' column
        self.mean_sarc_score = (round((self.df.prob_sarc2.mean()),2)*100)       # append this to the list of average scores when usernames evaluated
             
                # calculate median sarcasm score based on 'prob_sarc2' column
        self.median_sarc_score = (round((self.df.prob_sarc2.median()),2)*100)
                
                # sort the df by the scores  
        self.df = self.df.sort_values('prob_sarc2', ascending = False)
            
                # reset index to allow for accurate indexing of the most sarcastic score, etc.
        self.df = self.df.reset_index(drop=True)
            
                # Assign the df which is a method argument as a pipeline object attribute...
                    #.. to be used for the multinomial naive Bayes analysis
        self.df_nb = pd.DataFrame(words_lst)
        self.df_nb = self.df.rename(columns={0:'text_final'})
        
        #self.df_nb = df
            
                # Produce a prediction for each row of the 'text' column in 'self.df_nb'...
                    #... using multinomial naive Bayes
        self.df_nb['prob_sarc_nb'] = self.df_nb['text_final'].apply(lambda x: pipe_nb.predict_proba(x))  
            
                # create a new column for the probability which just has the score for sarcasm ...
                    #... created using multinomial naive Bayes
        self.df_nb['prob_sarc2_nb'] = self.df_nb['prob_sarc_nb'].apply(lambda x: x[0][1])
            
                # convert 'text' column with a single string in the list to a column consisting of list of words
        self.df_nb['text_words'] = self.df_nb['text_final'].apply(lambda x:[sentence.split() for sentence in x])
            
                # removes the nested list structure 
        self.df_nb['text_words'] = self.df_nb['text_words'].apply(lambda x: x[0])
            
                # Execute the multinomial naive Bayes pipeline on the 'text_words' column        
        self.df_nb['test_nb2'] = [pipe_nb.predict_proba(i) for i in self.df_nb['text_words']]
            
                # Create a new column with just the sarcasm score based on the ...
                    #... multinomial naive Bayes pipeline analysis     
        self.df_nb['mnb_sarc'] = self.df_nb['test_nb2'].apply(lambda x:[item[1] for item in x] )
        
            # sort the df by the scores  
        self.df_nb = self.df_nb.sort_values('mnb_sarc', ascending = False)
        
            # pickle the DF containing the word probabilities generated from the...
                #... multinomial naive Bayes model
        save_word_probs(self.df_nb)
        
        return(self.df, self.df_nb) 
        
        
        

if __name__ == '__main__':
    from pipeline_capstone2 import Make_Pipeline
    #test_tweet2 = "Naahhhh... My chart says it's a good time to buy. The ASX has already started to go back up so I've missed the bottom of the market dammit.How could mass unemployment, private and business defaults and recession affect the share market? They're not related."
    #test_tweet3 ="Hey @YouTube, thanks for keeping me from being able to create playlists of children's content on my TEACHER channel"
    #test_tweet4 ="Yes because letting everyone travel and continue to spread a virus is a great idea"
    #test_tweet5 =  ['cat', 'dog', 'monkey']  
    #tweet_predictor = Make_Pipeline.predict_tweet()
        # Create an instance of Class 'Make_Pipeline'
    pipe_test = Make_Pipeline() 
    #tweet_predictor.predict_tweet(test_tweet2)
    #pipe_test.predict_tweet("If the briefings do end Dr. Birx's scarves should get their own YouTube channel. That is all.","@marktraphagen") 
    #pipe_test.predict_tweet("@NateSilver538","@NateSilver538")
    #tweet_predictor = Make_Pipeline(test_tweet4)  
    #.predict_tweet()  
    #tweet_predictor.predict_tweet(test_tweet2)
        # Define an object which is an instance of Class 'Make_Pipeline' with the...
            #... 'predict_tweet' method    
    save_pipe = pipe_test.predict_tweet
    #save_pipe2 = pipe_test.pred_word_probs()
        # Pickle the object which isinstance of Class 'Make_Pipeline' containing the ...
            #... 'predict_tweet' method
    OOP_pickle_util.save_pipeline_object(save_pipe) 
    #print('Pipeline success!')
  
        
