# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:17:47 2020

@author: sinua
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:29:11 2020

@author: sinua

This file attempts to accomplish the following objective described by Ben:
    "make a preprocessor object with a fit method , that calls, in order, each
    of the other preprocessing method as steps and saves the output as a 
    attribute (self.whatever) "

Reference: https://towardsdatascience.com/deploying-models-to-flask-fb62155ca2c4
Reference: https://github.com/mmalinas/Springboard_Git/blob/master/Capstone2_MelanieM/website-final/class_def.py
"""

""" IMPORTS """

import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', None)
import re # for using regular expressions to remove numbers
import string # for removing punctuation
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import OOP_pickle_util
#import en_core_web_md
import en_core_web_sm
#nlp = en_core_web_md.load()
nlp = en_core_web_sm.load()
from io import StringIO
import tweepy as tw

    # guppy provides an estimate of memory use
from guppy import hpy
h = hpy()
print("Guppy and Heapy from 'Preprocessor.py'",h.heap())

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


US_names = ["US","U.S.","United States","United States of America", "America"]

US_locs = state_abbrevs + state_names + US_names # location data used to filter tweets 

    # specific words to remove from tweets. Most are indicators of sarcasm but 'has' ...
        #... is included because the lemmatizer turns it into 'ha'
bad_words = [ 'setupampfailed', 'aghe', 'nj', 'nt', 'ar', 'vo', 'bemies', 'em',
             'pre', 've', 'amp', 'ad', 'ha',
             'has','gonna', 'gotta', "don't", 'Sarcastic', 'sarcastic', 
             'Sarcasm','sarcasm', 'nsarcasm', 'nnsarcasm', '_', '__', '___',
             '____','______','___________________', '______________________']

    # dictionary of contractions
contractions = { 
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I had",
"I'd've": "I would have",
"I'll": "I I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

#Initialize Predictor class with csv files containing sarcastic tweets and
  #  non-sarcastic tweets from the same user names """         
    
class Preprocessor:
    def __init__(self, tweet_data):
        self.tweet_data = tweet_data
        self.lemmatizer = lemmatizer
            # Instantiate English stop words from nltk
        self.stop_words = set(stopwords.words('english')) 
        #self.nlp = en_core_web_sm.load()
        self.nlp = nlp
            
        # Import sarcastic and matching tweets, drop duplicates and filter for locations within the US 
    def import_tweets(self, csv_file): # import tweets csv
        df = pd.read_csv(csv_file) #reads all tweets
        df = df[df.location.str.contains('|'.join(US_locs), na=False)] # filter for US locations
        df = df.drop_duplicates(subset=['text','user','date']) # remove duplicate tweets / rows in the DF
        return df
        
        # converts tweet text column to string so regex will work
    def convert_to_string(self, df):
        df['text2'] = df['text'].apply(str)
        #df['text2'] = [' '.join(map(str, l)) for l in df['text']]
        return df
    
        # replace contractions with their non-contraction counterparts
    def remove_contractions(self, df):
        
            # first convert the tweet text from a string to a list 
        df['text3'] =df['text2'].apply(lambda x: x.split(' '))
            # replace contractions using the contractions dictionary
        df['text4'] = df['text3'].apply(lambda x:[contractions.get(item,item)  for item in x])
            #convert back to a string for further processing
        df['text5'] = [' '.join(map(str, l)) for l in df['text4']]
        #df['text6'] = df['text5'].apply(str)
        #df['text5'] = df['text4'].apply(str)
        return df
    
        # removes byte string indicators
    def remove_byte_strings(self, df):  
            # requires working column to be in string format for regex to work
        df['text6'] = df['text5'].str.replace('b\'|b\"',r'')
        #df['text5'] = [' '.join(map(str, l)) for l in df['text5']]
        #df['text5'] = df['text5'].apply(str)
        #df['text6'] = df['text5']
        return df
    
        # remove one of the common encoded patterns    
    def remove_encoded_patterns(self, df):
        #self.convert_to_string(df) # requires working column to be in string format for regex to work
        df['text7'] = df['text6'].str.replace(r"\\x.*? " ,r" ") 
        return df
 
        # remove another common encoded pattern 
    def remove_encoded_patterns_two(self, df): 
        #self.convert_to_string() # requires working column to be in string format for regex to work
        df['text8'] = df['text7'].str.replace(r"\\x.*?\'" ,r" ") 
        return df
    
     # remove another common encoded pattern: \n
    def remove_encoded_patterns_three(self, df): 
        #self.convert_to_string() # requires working column to be in string format for regex to work
        df['text9'] = df['text8'].str.replace(r"\\n" ,r" ") 
        return df
    
    # remove another common encoded pattern: \B
    def remove_encoded_patterns_four(self, df): 
        #self.convert_to_string() # requires working column to be in string format for regex to work
        df['text10'] = df['text9'].str.replace(r"\\B" ,r" ") 
        return df   
    
    # remove a common pattern: RT
    def remove_RT(self, df): 
        #self.convert_to_string() # requires working column to be in string format for regex to work
        df['text11'] = df['text10'].str.replace(r"RT " ,r" ") 
        return df
 
        # split the tweet text into a list of strings, needed for methods such as 'remove_bad_words'    
    def split_into_strings(self, df):    
         df['text12'] = df['text11'].apply(str)
        #df['text'] = df['text'].str.split() 
         return df
     
        # remove certain words such as those indicating sarcasm directly
    def remove_bad_words(self, df):
         # 'remove_bad_words" method requires a list of strings
        df['text13'] = df['text12'].str.split()
        #df['text7'] = df['text6'].apply(str)
        df['text14'] = df['text13'].apply(lambda x: [word for word in x if word not in bad_words]) 
        return df        
     
        # lowercase the tweet text        
    def lowercase(self, df):  
        df['text15'] = df['text14'].apply(lambda x: [word.lower() for word in x]) 
        return df
  
        # remove numbers from the tweet text. Works on either a list or a string
    def remove_numbers(self, df):
         df['text16'] = df['text15'].astype(str) # convert text within working columns to string first
         df['text17'] = df['text16'].str.replace('\d+', '')
         return df 

        # remove usernames from tweet text.
    def remove_usernames(self, df):
             # formats working column as list of strings which enables the lamda function to work 
         df['text18'] = df['text17'].str.split() 
         df['text19'] = df['text18'].apply(lambda x: re.sub(r'(^|[^@\w])@(\w{1,15})\b', '', str(x)))
         return df
  
        # remove hashtags from tweet text
    def remove_hashtags(self, df):
             # formats working column as list of strings which enables the lamda function to work 
        df['text20'] = df['text19'].str.split() 
        df['text21'] = df['text20'].apply(lambda x: re.sub(r'(^|[^#\w])#(\w{1,50})\b', '', str(x)))
        return df
  
        # remove punctuation
    def remove_punctuation(self, df):    
         df['text22'] = df['text21'].str.replace(r'[^\w\s]+', '') 
         return df
     
        # remove urls from tweet text; seems to be robust to various formats of the working column
    def remove_urls(self, df):  
         df['text23'] = df['text22'].replace(r'http\S+', '', regex=True)
         return df
              
        # lemmatize tweet text
    def lemmatize(self, df):
         #from nltk.stem import WordNetLemmatizer
         #lemmatizer=WordNetLemmatizer() # replaced lemmatizer below with self.lemmatizer
         df['text24'] = df['text23'].str.split()
             #split_into_strings() # split into list of strings, needed ...
             #... for the lemmatizer to operate on individuals words rather than individual characters
         df['text25'] = df['text24'].apply(lambda x: [self.lemmatizer.lemmatize(word) for word in x]) # lemmatize
         return df
   
        # remove stopwords
    def remove_stopwords(self, df): 
         #from nltk.corpus import stopwords
         #nltk.download('stopwords')
             # Instantiate English stop words from nltk
         #stop_words = set(stopwords.words('english'))  # using self.stopwords below instead
             #split_into_strings() # split into list of strings, needed ...
             #...to operate on individuals words rather than individual characters
         df['text26'] = df['text25'].apply(lambda x: [word for word in x if word not in self.stop_words])
         df['text27'] = [' '.join(map(str, l)) for l in df['text26']]
         return df
        
        #  removes proper nouns from the tweet text column using spaCy and
        #  converts the working column of tweet text back to string format which is required by CountVectorizer 
    def remove_proper_nouns(self, df):
        #import en_core_web_md
        #nlp = en_core_web_md.load()    
        print('remove_proper_nouns started')
        df['text28'] = df['text27'].apply(lambda x: self.nlp(x))
        df['text29'] = df['text28'].apply(lambda x: [word for word in x if word.pos_ != 'PROPN'])
        df['text30'] = [' '.join(map(str, l)) for l in df['text29']] # returns the working column of processed tweet text data to string format required by CountVectorizer 
        print('remove proper nouns finished')
        return df
    
     # remove any remaining nonsense words left over and/or created as a result of previous preprocessing steps
    def remove_bad_words2(self, df):
         # 'remove_bad_wordss" method requires a list of strings
        print('remove_bad_words2 started')
        df['text31'] = df['text30'].str.split()
        #df['text7'] = df['text6'].apply(str)
        df['text32'] = df['text31'].apply(lambda x: [word for word in x if word not in bad_words]) 
        df['text33'] = [' '.join(map(str, l)) for l in df['text32']] # returns the working column of processed tweet text data to string format required by CountVectorizer   
        print('remove bad_words2 ending')
        print("debugging df:",df.info)
        return df  
    
        # Remove tweets related to politics using spaCy's semantic similarity scores and create a new working
        #  ...DF containing tweets less-related to politics
    def remove_pol_tweets(self, df):
            # Create a word embedding using spaCy for words related to politics
        political = self.nlp('politics elections vote government congress senate president Trump potus flotus scotus democrat democrats republican republicans')
            # Create an empty list to store word embedding scores related to politics
        scores = []    
            # Iterate over the preprocessed text for each tweet, create a word embedding, 
            # ...score each tweet for politics, and append scores to a list
        for i, text in enumerate(df['text33']): #consider using apply
            word_sim = self.nlp(text)
            score = word_sim.similarity(political)
            scores.append(score) 
        print("# columns before inserting political score as a new column", len(df.columns))             
            # Insert the word embedding scores for politics as a new column
            # programmatically insert at position two less than the # columns
        new_col_pos = (len(df.columns)-1)
        df.insert(new_col_pos, "political_score", scores, True) # insert 'political_score' as the second to last column   
            #df.insert(0, "political_score", scores, True)
        df = df[df['political_score'] < 0.5] # new working DF with politcal tweets removed
            # copies the 'tweet_category' columns and places the copy at the end of the DF...
            # needed to retain this data which becomes the target array
        print("column names:",df.columns)
        """
        if 'tweet_category' in list(df.columns):
            print("DF contains label column 'tweet_category' so data probably from a .csv")
            df['tweet_cat'] = df['tweet_category']
        else:
            print("DF lacks label column 'tweet_category' so data probably from a username")
        """
        # copies the original tweet text column to the end of the DF so it...
                #... is retained for comparing to the preprocessed text
        df['original_tweet'] = df['text']
        print("# columns before editing", len(df.columns))
        df = df.iloc[:,new_col_pos:] # filter the df to only retain the last three columns which is important for preventing a pickling error because spacy does not support pickling tokens
        return df
  
    # Remove tweets related to politics using spaCy's semantic similarity scores...
    #...and create a new working...DF containing tweets less-related to politics...
    #... for usernames. Differs from the method above because the returned DF...
    #... includes a column for tweet_id which is needed for Flask and the webpage...
    #... to use JavaScript for displaying the Twitter tweet widget 
    def remove_pol_tweets_username(self, df):
            # Create a word embedding using spaCy for words related to politics
        political = self.nlp('politics elections vote government congress senate president Trump potus flotus scotus democrat democrats republican republicans')
            # Create an empty list to store word embedding scores related to politics
        scores = []    
            # Iterate over the preprocessed text for each tweet, create a word embedding, 
            # ...score each tweet for politics, and append scores to a list
        for i, text in enumerate(df['text33']): #consider using apply
            word_sim = self.nlp(text)
            score = word_sim.similarity(political)
            scores.append(score) 
        print("# columns before inserting political score as a new column", len(df.columns))             
            # Insert the word embedding scores for politics as a new column
            # programmatically insert at position one less than length of # columns
        new_col_pos = (len(df.columns)-5)
        df.insert(new_col_pos, "political_score", scores, True) # insert 'political_score' as the second to last column   
            #df.insert(0, "political_score", scores, True)
        df = df[df['political_score'] < 0.5] # new working DF with politcal tweets removed
            # rename the processed tweet column so the new column has a ...
            #... expected by the 'predict_tweet' method of 'pipeline_capstone2.py'
        df = df.rename(columns={"text26": "text"})
        # copies the 'id' column to the end so it is not removed. These data needed...
            #... for the Twitter tweet widget
        df['tweet_ids']  =df['ids']
        print("# columns before editing", len(df.columns))
        df = df.iloc[:,new_col_pos:] # filter the df to only retain the last two columns which is important for preventing a pickling error because spacy does not support pickling tokens
        print("# columns after editing", len(df.columns))
        return df
    
        # eliminates tweets with zero words remaining and removes 'words' consisting of a single letter
    def remove_empty_tweets(self, df):
        df['text34'] = df['text33'].str.split() # converts column 'text33' from string to a list
        df['text35'] = df['text34'].apply(lambda x:[i for i in x if len(i) >1])# remove 'words' consisting of a single letter
        df['text36'] = df['text35'].apply(lambda x: len(x)) # count words in 'text34'
        df = df[df['text36'] > 0]    # retain rows with more than zero words
        df['text_final'] = [' '.join(map(str, l)) for l in df['text35']] # returns the working column of processed tweet text data to string format required by CountVectorizer   
        return df
    
   #  The method below implements the above methods in a sequence when the input is a .csv file....
   # ... as occurs for entering training data    

    def fit_preprocess_methods(self, data):
       
        tweet_data = self.import_tweets(data)
        s_data = self.convert_to_string(tweet_data)
        remove_contractions = self.remove_contractions(s_data)
        rem_bytes = self.remove_byte_strings(remove_contractions)
        rem_enc = self.remove_encoded_patterns(rem_bytes)
        rem_enc2 = self.remove_encoded_patterns_two(rem_enc)
        rem_enc3 = self.remove_encoded_patterns_three(rem_enc2)
        rem_enc4 = self.remove_encoded_patterns_four(rem_enc3)
        rem_rt = self.remove_RT(rem_enc4)
        s_data2 = self.split_into_strings(rem_rt)
        no_bads = self.remove_bad_words(s_data2)
        lower = self.lowercase(no_bads)
        no_nums = self.remove_numbers(lower)
        no_users = self.remove_usernames(no_nums)
        no_hash = self.remove_hashtags(no_users)
        no_punc = self.remove_punctuation(no_hash)
        no_urls = self.remove_urls(no_punc)
        lem = self.lemmatize(no_urls)
        no_stops = self.remove_stopwords(lem)
        no_pns = self.remove_proper_nouns(no_stops)
        print('finished remove_proper_nouns')
        no_bads2 = self.remove_bad_words2(no_pns) # clean up any remaining nonsense words left over from preprocessing
        #print("debugging df:",self.df.info)
        no_pols = self.remove_pol_tweets(no_bads2)
        no_zeroes = self.remove_empty_tweets(no_pols)
        print('finished remove_pol_tweets')
         
        self.df = no_zeroes
        return self.df # accessable by calling >>>df = <class_name>.fit_preprocess_methods() ....
            #...and verified by calling >>><class_name>.__dict__.keys() ; see line 85 in 'OOP_clas_def.py'
    
   # The method below implements the same methods as 'def fit_preprocess_methods'...
   #     ... above except not 'def tweet-data" in order to handle a DF such as that returned ...
    #    ... by 'def handle_username'.    

    def fit_preprocess_methods_df(self, df):
       
        #tweet_data = self.import_tweets(data)
        s_data = self.convert_to_string(df)
        remove_contractions = self.remove_contractions(s_data)
        rem_bytes = self.remove_byte_strings(remove_contractions)
        rem_enc = self.remove_encoded_patterns(rem_bytes)
        rem_enc2 = self.remove_encoded_patterns_two(rem_enc)
        rem_enc3 = self.remove_encoded_patterns_three(rem_enc2)
        rem_enc4 = self.remove_encoded_patterns_four(rem_enc3)
        rem_rt = self.remove_RT(rem_enc4)
        s_data2 = self.split_into_strings(rem_rt)
        no_bads = self.remove_bad_words(s_data2)
        lower = self.lowercase(no_bads)
        no_nums = self.remove_numbers(lower)
        no_users = self.remove_usernames(no_nums)
        no_hash = self.remove_hashtags(no_users)
        no_punc = self.remove_punctuation(no_hash)
        no_urls = self.remove_urls(no_punc)
        lem = self.lemmatize(no_urls)
        no_stops = self.remove_stopwords(lem)
        no_pns = self.remove_proper_nouns(no_stops)
        print('finished remove_proper_nouns')
        no_bads2 = self.remove_bad_words2(no_pns) # clean up any remaining nonsense words left over from preprocessing
        print("debugging df:", df.info)
        no_pols = self.remove_pol_tweets_username(no_bads2) 
        print('finished remove_pol_tweets_username')
        no_zeroes = self.remove_empty_tweets(no_pols)
        self.df = no_zeroes
        return self.df # accessable by calling >>>df = <class_name>.fit_preprocess_methods() ....
            #...and verified by calling >>><class_name>.__dict__.keys() ; see line 85 in 'OOP_clas_def.py'
    
    

   # Preprocesses a single string, as when evaluating a single tweet 
    
    def preprocess(self, tweet_data): 
            
        text2 = tweet_data.replace(r"b\'|b\"",r"") # remove the byte string indicators
        text3 = text2.replace(r"\\x.*? " ,r" ") # remove one of the common encoded patterns
        text4 = text3.replace(r"\\x.*?\'" ,r" ") # remove another common encoded pattern
        text5 = text4.replace(r"\n" ,r" ") # remove another common encoded pattern
        split_tweet = text5.split() # split the tweet text into a list of strings 
        new_words = split_tweet
        for word in split_tweet:   
            if word in bad_words:
                new_words.remove(word) # remove certain words 
        words_low = [word.lower() for word in new_words] # lowercase words
        no_nums = [i for i in words_low if not i.isdigit()] # remove digits
        no_users = [a for a in no_nums if not re.search('(^|[^@\w])@(\w{1,15})\b',a)]  # remove names  
        no_hash = [a for a in no_users if not re.search('(^|[^#\w])#(\w{1,15})\b',a)]   # remove hashtags
        no_punc = [a for a in no_hash if not re.search('[^A-Za-z0-9]+',a)] # remove punctuation 
        no_urls =  [a for a in no_punc if not re.search('^http',a)] # remove urls 
        lem_tweets = [lemmatizer.lemmatize(word) for word in no_urls] # lemmatize
        no_stops = [word for word in lem_tweets if word not in stopwords.words('english')] # remove stopwords
        processed = ' '.join(no_stops) # convert the tweet text back into a string to be evaluated by the model
        df_column_title_and_tweet = 'text\n' + processed
        df_contents = StringIO(df_column_title_and_tweet)
        self.df = pd.read_csv(df_contents)    
        
        self.df['original_tweet'] = tweet_data
                
        return self.df 
    
    # 'def handle_list' takes a list of tweet texts and iteratively calls the ...
    #.... 'def preprocess' method on each tweet text in turn 
    # since each tweet is coverted into a df, it must have a concat component...
    #... so the final return value is a single df
    
    def handle_list(self, list_data): #changed argument to 'list_data' from 'tweet_data' to avoid potential confusion
        print('starting def handle_list')
            # list to collect dfs created by the 'preprocess' method as each ...
            # ... original tweet in 'list_data' is processed and transformed into...
            #... a df of length one or length zero
        self.dfs = []
        
            # this list accumulates the length of dfs after preprocessing in order to...
            #... track which ones have a length of zero so their indices can be identified
            #... and the indices can be used to eliminate those tweets from the original...
            #... list so the original tweets can be retained in the df returned by this method
       # length_dfs = [] # code no longer needed due to edits in the 'preprocess' method
        
        for item in list_data:
                # call the 'preprocess' method on each original tweet in list_data
                # each original tweet is preprocessed as a string and a df of length one...
                #... or length zero is returned
            self.preprocess(item)
            
                # append the length of each df resulting from preprocessing. The ...
                # ... possible values are zero and one. Those with values of zero...
                #...  are relavent to retaining original tweets in the ...
                # df which is finally returned after a username is processed
            #length_dfs.append(len(self.df)) # code no longer needed due to edits in the 'preprocess' method
            
                # append each df resulting from calling the 'preprocess' method...
                #... on each original tweet in 'tweet_data'
            self.dfs.append(self.df)
        
        # print("length_dfs: ", length_dfs) # code no longer needed due to edits in the 'preprocess' method
        
            # accumulates indices that identify tweets that result in dfs with ...
            #... a length of zero which is needed for retaining original tweets
       # zero_df_indices = [] # # code no longer needed due to edits in the 'preprocess' method
        #for i, j in enumerate(length_dfs): # code no longer needed due to edits in the 'preprocess' method
         #   if j == 0: # code no longer needed due to edits in the 'preprocess' method
          #      zero_df_indices.append(i) # code no longer needed due to edits in the 'preprocess' method
        
            # creates list of original tweets that do not result in dfs with a length ...
            #... of zero due to the 'preprocess' method. Needed for adding the original...
            #... tweets as a column to the df which is returned and making sure the new...
            #... column has the same length as the df
            #reference: https://stackoverflow.com/questions/18044032/in-place-function-to-remove-an-item-using-index-in-python
            
     #   non_zero_tweets = [x for i, x in enumerate(list_data) if i not in zero_df_indices] # # code no longer needed due to edits in the 'preprocess' method
     #   print("length of original list of tweets after removing those resulting in a df with length zero: : ", len(non_zero_tweets)) # code no longer needed due to edits in the 'preprocess' method
       
            # concatenate the the list of dfs created by the 'preprocess method'...
            #... as a result of calling this method on the original list of tweets - 'tweet_data'
        self.df = pd.concat(self.dfs)
            
            # Include original tweet data as a new column 
       # self.df['original_tweet'] = non_zero_tweets # code no longer needed due to edits in the 'preprocess' method
        
        return self.df
        
    
    def handle_short_tweet(self, tweet_data):
        print('handling very short entry')
        short_tweet_sub = "Please enter a longer tweet"
        self.preprocess(short_tweet_sub) 
        return self.df
        
        # grab up to 100 tweets for a single username using the twitter api and...
        #..return them as a list 'username_tweets' 
        
    def handle_username(self, tweet_data):
            #remove the @ symbol from the username
        tweet_data = re.sub(re.compile('@'),"",tweet_data)
          
            # twitter api code from "Capstone2_Draft1.ipnyb'
            # CAUTION: DO NOT SHARE THIS INFORMATION
            # Personal API Keys from the Twitter App page
            # keys saved in 'Twitter_API_Keys.txt' and in B-folder
        access_token =  
        access_token_secret =  
        consumer_key =  
        consumer_secret =          
            # OAuth process, using the keys and tokens
        auth = tw.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
            # Create the api to connect to twitter with your credentials; creation...
            #...of the actual interface, using authentication
            # previous api call version resulting in rate limit issues
            #api = tw.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True) 
            # new version of api call based on Sam Gunnerson's code 
        api = tw.API(auth,parser=tw.parsers.JSONParser())
            
            # gather 100 new tweets 
        self.new_tweets = api.user_timeline(screen_name = tweet_data, count=100, include_rts=False)
            # add the new tweets to the list - needed?
        #self.username_tweets = self.username_tweets.extend(self.new_tweets)
            # create a list of the 'text' column from the new tweets
        self.outtweets = [tweet['text'] for tweet in self.new_tweets]
        
            # create a new list of the 'id' column from the new tweets
        self.out_ids = [tweet['id'] for tweet in self.new_tweets]
       
            # combine 'outtweets' and 'out_ids' into a DF
        self.username_df = pd.DataFrame(list(zip(self.out_ids, self.outtweets)), columns = ["ids","text"])
        print("debug username_df", self.username_df.info())
        self.fit_preprocess_methods_df(self.username_df)
        
        return self.outtweets, self.out_ids, self.username_df, self.df
    
  #  Checks the format of input tweet data and calls the appropriate method 
  #      above - either 'def preprocess' or 'def fit_preprocess_methods' depending 
  #      on the input format - either a string or .csv, respectively  
    
    def tweet_format(self):
        if type(self.tweet_data) == list: #processing a list is necessary for ...
            #... analyzing a username's tweets because 'def handle_username' ...
            #... returns a list 
            self.handle_list(self.tweet_data) 
            print('list processed')
        
        elif self.tweet_data.endswith('.csv'): # applies for training the model 
            print(True)
            self.fit_preprocess_methods(self.tweet_data) # processes a whole .csv file and creates a DF
            print('csv processed')
            
        # elif statement that checks for a blank or short string
        elif len(self.tweet_data) < 1:    
            self.handle_short_tweet(self.tweet_data) # address very short entries
            print("Handling a very short entry")
            self.preprocess
         
        # check for username
        elif self.tweet_data.startswith('@'):
            print("username entered")
            self.handle_username(self.tweet_data)
            print("Username processed") 
            
           # processes a single tweet / string     
        else: 
            print(False)
            self.preprocess(self.tweet_data) # processes a single tweet and creates a DF
            print('string processed...')

  # Note the current configuration of the main function evaluates the full training
  #      dataset and will take awhile to complete  
        





if __name__ == '__main__':
    from Preprocessor import Preprocessor
     
        #preprocessor instance using the full training dataset .csv. This is a ...
            #... needed first step for creating the model before calling the ...
                #... 'traind_and_fit' method of 'OOP_clas_def.py' which creates ...
                    #... and pickles the model which can then be used in the ...
                        #... pipeline 
    #pre_inst_s = Preprocessor('all_sarc_and_matching_tweets.csv') # preprocess the sarcasm data
    #prepro_instance = Preprocessor('kag_data_small.csv') # preprocess the Kaggle Twitter data after the sentiment140 data is reduced to a size that matches the preprocessed sarcasm data (~48,000 rows)
    prepro_instance2 = Preprocessor('test_data.csv')
    #prepro_instance3 = Preprocessor('@DarthVader')
    #preprocessor_instance = Preprocessor('@CodeNewbies')
        # call the 'tweet_format' method to initiate data processing
    #preprocessor_instance.tweet_format() # call the 'tweet_format' method to initiate data processing
    #pre_inst_s.tweet_format() # call the 'tweet_format' method to initiate data processing on the sarcasm data preprocessor instance
    #prepro_instance.tweet_format() # call the 'tweet_format' method to initiate data processing on the Kaggle Twitter data preprocessor instance
    prepro_instance2.tweet_format()
    # pickle the preprocessor instance including attributes such as the processed...
            #...tweet DF. This a needed second step for creating the model before...
                #... calling the 'train_and_fit' method of 'OOP_clas_def.py' which...
                    #... creates and pickles the model which can then be used...
                            #... in the pipeline
    #OOP_pickle_util.save_sent_prepro_inst(prepro_instance) # pickle the Kaggle Twitter preprocessor instance 
    #OOP_pickle_util.save_preprocessor_instance(pre_inst_s) # pickle the sarcasm data preprocessor instance
        #use to test ability to process a single tweet
    #test_tweet = 'a bunch of random words to process and test the new changes'
    #test_tweet2 = '@DarthVader'
    print('Preprocessor success!')
    #profile.run()
 





