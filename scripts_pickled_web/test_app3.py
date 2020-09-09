from flask import Flask, render_template, request
import OOP_pickle_util
from OOP_pickle_util import load_pipeline_object
import logging
from logging.handlers import RotatingFileHandler
from logging.handlers import SMTPHandler 
import smtplib # required for email 
from email.message import EmailMessage # required for email
from email.mime.multipart import MIMEMultipart # required for email
from email.mime.base import MIMEBase # required for email
from email import encoders
from email.mime.text import MIMEText

    # guppy provides an estimate of memory use
#import guppy
#from guppy import hpy
#h = hpy()
#print("Guppy and Heapy from 'test_app3.py",h.heap())

pipe_object = load_pipeline_object('pipeline_object_pickled')

    # load the largest average score for a username
#largest_ave_score =  OOP_pickle_util.load_largest_ave_score('largest_ave_score_pickled')        
    # load the username with the largest average tweet score
#largest_username =  OOP_pickle_util.load_largest_username('largest_username_pickled') 
    # load the lowest average tweet score
#lowest_ave_score =  OOP_pickle_util.load_lowest_ave_score('lowest_ave_score_pickled') 
    # load the username with the lowest average tweet score
#lowest_username =  OOP_pickle_util.load_lowest_username('lowest_username_pickled')     
    # load the largest single tweet score
#largest_score =  OOP_pickle_util.load_largest_score('largest_score_pickled')  
    # load the tweet with the largest single tweet score
#largest_tweet =  OOP_pickle_util.load_largest_tweet('largest_tweet_pickled') 
    # load the lowest single tweet score 
#lowest_score =  OOP_pickle_util.load_lowest_score('lowest_score_pickled') 
        
#lowest_tweet =  OOP_pickle_util.load_lowest_tweet('lowest_tweet_pickled') 

    # load pickled list of average scores
average_scores = OOP_pickle_util.load_average_scores('average_scores_pickled')

    # calculate the average score across all usernames
total = 0
for score in average_scores:
    total = total + score
ave_username_score = round(total/len(average_scores),2)

#mail = Mail()

app = Flask(__name__) # original working version this version works on my local machine 


@app.route('/about3')
def about():
    return render_template('about3.html')

#app = Flask(__name__, static_url_path="/static", static_folder='C:/Users/sinua/Documents/Data_Science_Coding/Springboard/Capstone2/static') # version suggested by reference: https://stackoverflow.com/questions/44340416/css-and-js-not-working-on-flask-framework
#app = Flask(__name__,
#            static_folder='static',
#            template_folder='Templates') # reference: https://stackoverflow.com/questions/58269486/how-to-get-flask-recognise-the-css-and-js-files-in-the-static-folder
#app = Flask(__name__, static_url_path='/static') # reference: https://stackoverflow.com/questions/54923070/flask-javascript-and-css-not-being-correctly-rendered

#app = Flask(name, template_folder='/home/williamwebb35/templates') # version suggested by...
    #... pythonanywhere forums discussion to help solve pythonanywhere 'templateNotFound' error ; reference: https://www.pythonanywhere.com/forums/topic/1039/

    # testing 'test7.html' for the vertical line with a horizontal bar chart...making...
    #... sure the template works with the previous files
  
@app.route('/', methods=['GET', 'POST'])
def test7():        
    if request.method == "GET":
              
        #return render_template('test7.html') #working version w/o carousel
        #return render_template('index9.html') # working version with widget carousel 
        #return render_template('index10.html') # working version with widget carousel, testing barchart carousel
        #return app.send_static_file('index9.html')     # version suggested by reference: https://stackoverflow.com/questions/44340416/css-and-js-not-working-on-flask-framework                    
        #return render_template('owl_carousel_test.html')
        return render_template('index15.html') # draft version with synced carousel
        
    else:
    #elif request.form['novel_username'].startswith('@'):
    #username = x #test value
        # names for the objects returned by the 'predict_tweet' method of 'pipeline_capstone2.py'...
        #... which correspond to the following objects returned by 'predict_tweet':...
        #...tweet_data, prob_sarc_tweet, self.df, ave_sarc_score, self.mean_sarc_score, self.median_sarc_score
    
        #app.logger.warning('A warning occurred (%d apples)', 42)
        #app.logger.error('An error occurred')
        #app.logger.info('Info')
        
        x = request.form['novel_username']    
        y = str(x) 
        
        if y.startswith('@'):
            
            username = request.form['novel_username'] 
            
                # pipeline object using the entered novel username
            x_input, predictions, df, df_nb, ave_sarc_score, mean_sarc_score, median_sarc_score,most_sarcastic_tweet, most_sarcastic_score= pipe_object(y, username)
            
                # tweet id for the novel username's most sarcastic tweet
            most_sarcastic_id0 = df.tweet_ids[0]
                # tweet id for the novel username's 2nd most sarcastic tweet
            most_sarcastic_id1 = df.tweet_ids[1]
                # tweet id for the novel username's 3rd most sarcastic tweet
            most_sarcastic_id2 = df.tweet_ids[2]
                # tweet id for the novel username's 4th most sarcastic tweet
            most_sarcastic_id3 = df.tweet_ids[3]
                # tweet id for the novel username's 5th most sarcastic tweet
            most_sarcastic_id4 = df.tweet_ids[4]
            
                # sarcasm probability score for the novel username's most sarcastic tweet
            prob_sarc2_0 = round(df.prob_sarc2[0]*100)
                # sarcasm probability score  for the novel username's 2nd most sarcastic tweet
            prob_sarc2_1 = round(df.prob_sarc2[1]*100) 
                # sarcasm probability score  for the novel username's 3rd most sarcastic tweet
            prob_sarc2_2 = round(df.prob_sarc2[2]*100) 
                # sarcasm probability score  for the novel username's 4th most sarcastic tweet
            prob_sarc2_3 = round(df.prob_sarc2[3]*100) 
                # sarcasm probability score  for the novel username's 5th most sarcastic tweet
            prob_sarc2_4 = round(df.prob_sarc2[4]*100)
                #scores for the second barchart located in the carousel
            scores = [prob_sarc2_0, prob_sarc2_1, prob_sarc2_2, prob_sarc2_3, prob_sarc2_4]
            
                # round the value for the most sarcastic score
            most_sarc_score = round(most_sarcastic_score,2)*100
            
                # add additional celebrity accounts here with additional calls to ...
                    #... the pipeline
                # pipeline for @BarackObama
            BA_x_input, BA_predictions, BA_df, BA_df_nb, BA_ave_sarc_score, BA_mean_sarc_score, BA_median_sarc_score, BA_most_sarcastic_tweet, BA_most_sarcastic_score = pipe_object("@BarackObama", "@BarackObama")
                #pipeline for @katyperry
            KP_x_input, KP_predictions, KP_df, KP_df_nb, KP_ave_sarc_score, KP_mean_sarc_score, KP_median_sarc_score, KP_most_sarcastic_tweet, KP_most_sarcastic_score = pipe_object("@katyperry", "@katyperry")
                #pipeline for @DarthVader
            DV_x_input, DV_predictions, DV_df, DV_df_nb, DV_ave_sarc_score, DV_mean_sarc_score, DV_median_sarc_score, DV_most_sarcastic_tweet, DV_most_sarcastic_score = pipe_object("@DarthVader", "@DarthVader")
                    
            
            names = [username,"@BarackObama","@katyperry", "@DarthVader"]#,"Average Username Score"] 
                #bar_labels = ['a','b','c','d'] # fixed values for testing
            scores = [ave_sarc_score, BA_ave_sarc_score, KP_ave_sarc_score, DV_ave_sarc_score]#, ave_username_score]
                #bar_values = [10,20,30,40] # fixed values for testing
                # create dict from names and scores for sorting scores descending
            name_scores = dict(zip(names, scores))
                # create a list of tuples sorted by scores descending for sorting scores descending
            sort_scores = sorted(name_scores.items(), key=lambda x: x[1], reverse=True)
                # convert sorted list of tuples into a dictionary for sorting scores descending
            sort_scores = dict(sort_scores)
                # create a list of scores now sorted descending
            bar_values = list(sort_scores.values())
                # create a list of usernames sorted according to respective scores, descending
            bar_labels = list(sort_scores.keys())
            
                # code block for organizing the labels and scores of the first...
                    #... barchart in the third carousel
                # labels for the first barchart of the third carousel
            """
            carouselthree_labels0 = df_nb.text[0]
                # values for the first barchart of the third carousel 
            carouselthree_values0 = df_nb.mnb_sarc[0]
                # create dict from labels and values for sorting values descending
            labels_values0 = dict(zip(carouselthree_labels0, carouselthree_values0))
                # create a list of tuples sorted by values descending for sorting values descending
            sort_labels0 = sorted(labels_values0.items(), key=lambda x: x[1], reverse=True)
                # convert sorted list of tuples into a dictionary for sorting values descending
            sort_labels0 = dict(sort_labels0)
                # create a list of values now sorted descending
            car_values0 = list(sort_labels0.values())
                # create a list of usernames sorted according to respective scores, descending
            car_labels0 = list(sort_labels0.keys())
            """
            car_labels0 = df_nb.text_final[0][0].split()
            car_labels1 = df_nb.text_final[1][0].split()
            car_labels2 = df_nb.text_final[2][0].split()
            car_labels3 = df_nb.text_final[3][0].split()
            car_labels4 = df_nb.text_final[4][0].split()
            
            car_values0 = sorted(df_nb.mnb_sarc[0],reverse = True)
            car_values1 = sorted(df_nb.mnb_sarc[1],reverse = True)
            car_values2 = sorted(df_nb.mnb_sarc[2],reverse = True)
            car_values3 = sorted(df_nb.mnb_sarc[3],reverse = True)
            car_values4 = sorted(df_nb.mnb_sarc[4],reverse = True)
            
            #bar_labels=labels # original comment out for testing
            #bar_values=values # original comment out for testing
                
            #return render_template('test7.html',           # working version w/o carousel
            #return render_template('index9.html',           # working version with widget carousel
            #return render_template('owl_carousel_test.html')
            #return render_template('index10.html', # working version with widget carousel and testing barchart carousel
            return render_template('index15.html', # draft version with synced carousel                          
                                       #username = username,
                                       #ave_sarc_score = ave_sarc_score,
                                       #BA_ave_sarc_score = BA_ave_sarc_score,
                                       #title='Username Score Comparison', 
                                       #max=100, 
                                       bar_labels=bar_labels, #original, comment out for testing
                                       #bar_values=bar_values,
                                       bar_values=bar_values,
                                       ave_username_score = ave_username_score,
                                       embed_tweet = "https://twitter.com/NateSilver538/status/1260324940114464768",
                                       #tweet_id = '899138450657484800',
                                       most_sarcastic_id0 = most_sarcastic_id0,
                                       most_sarcastic_id1 = most_sarcastic_id1,
                                       most_sarcastic_id2 = most_sarcastic_id2,
                                       most_sarcastic_id3 = most_sarcastic_id3,
                                       most_sarcastic_id4 = most_sarcastic_id4,
                                       prob_sarc2_0 = prob_sarc2_0,
                                       prob_sarc2_1 = prob_sarc2_1,
                                       prob_sarc2_2 = prob_sarc2_2,
                                       prob_sarc2_3 = prob_sarc2_3,
                                       prob_sarc2_4 = prob_sarc2_4,
                                       most_sarc_score = most_sarc_score,
                                       scores = scores,
                                       username = username,
                                       df_nb = df_nb,
                                       #car_values0 = car_values0,
                                       car_labels0 = car_labels0,
                                       car_labels1 = car_labels1,
                                       car_labels2 = car_labels2,
                                       car_labels3 = car_labels3,
                                       car_labels4 = car_labels4,
                                       car_values0 = car_values0,
                                       car_values1 = car_values1,
                                       car_values2 = car_values2,
                                       car_values3 = car_values3,
                                       car_values4 = car_values4)
                                        # tweet id for the most-sarcastic tweet, which should be...
                                                                #... in the first row"""
        
        else:
            #return render_template('test7.html') #working version w/o carousel
            #return render_template('index9.html') # working version with widget carousel
            #return render_template('index10.html') # working version with widget carousel, testing barchart carousel
            #return app.send_static_file('index9.html')     # version suggested by reference: https://stackoverflow.com/questions/44340416/css-and-js-not-working-on-flask-framework                    
            #return render_template('owl_carousel_test.html')
            return render_template('index15.html') # draft version with synced carousel

    # Catches an internal server error and returns the homepage
    # Creates an error log in case of an internal server error
"""
@app.errorhandler(500)
def page_not_found(e):
    
    # code block below creates log file
    app.logger = logging.getLogger('dev')
    app.logger.setLevel(logging.DEBUG)
        # full error log returned when file give '.log' suffix and just the ...
            #... offending username if the file is given '.txt' suffix below...
                #... and the other lines where it is referenced below 
    fileHandler = RotatingFileHandler('new_test.txt', maxBytes=10000, backupCount=1)
    fileHandler.setLevel(logging.DEBUG)
    app.logger.addHandler(fileHandler)
    app.logger.error('AN INTERNAL ERROR OCCURRED')
    x = request.form['novel_username']
    app.logger.error('USERNAME CAUSING THE ERROR IS %s',x)
"""
"""
        # generates an email with the log file attached
    sender_email =  
    receiver_email =  
    msg = MIMEMultipart()
    msg['Subject'] = '[Error Log Email]'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    #filename = "new_test.log"
    part = MIMEBase('application', "octet-stream")
    part.set_payload(open("new_test.txt", "rb").read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="new_test.txt"')
    msg.attach(part)
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as smtpObj:
            smtpObj.ehlo() 
            smtpObj.starttls()
            smtpObj.login( )
            smtpObj.send_message(msg)
    except Exception as e:
        print(e)
    
        # returns the homepage even if the username doesn't work
    return render_template("index15.html")
"""

    
if __name__ == "__main__":
    app.run()
    #handler = RotatingFileHandler('foo.log', maxBytes=10000, backupCount=1)
    #handler.setLevel(logging.INFO)
    #app.logger.addHandler(handler)
    """
        # code block below creates log file
    app.logger = logging.getLogger('dev')
    app.logger.setLevel(logging.DEBUG)
    #fileHandler = logging.FileHandler('new_test.log')
    fileHandler = RotatingFileHandler('new_test.log', maxBytes=10000, backupCount=1)
    fileHandler.setLevel(logging.DEBUG)
    app.logger.addHandler(fileHandler)

    
        # send an email using smtplib
        # code blocks below inspired by https://www.youtube.com/watch?v=sXjpkcF7rPQ and..
        #... https://docs.python.org/3/library/email.examples.html and ...
        #... https://stackoverflow.com/questions/9541837/attach-a-txt-file-in-python-smtplib
    
    msg = EmailMessage()
    msg['Subject'] = 'New Email from test_app3.py'
    
    sender_email = " "
    rec_email = " "
    password = " "
    message = "sent using python from flask within test_app3.py"
    
    msg['From'] = sender_email
    msg['To'] = rec_email
    with open('test_output.txt', 'rb') as fp:
        log_data = fp.read()
    msg.add_attachment(log_data, maintype = 'text', subtype = 'text' )
    
    
        # send the email message using smtplib
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, password)
    print("Login success")
    #server.sendmail(sender_email, rec_email, message)
    server.send_message(msg)
    """
 
    
   
