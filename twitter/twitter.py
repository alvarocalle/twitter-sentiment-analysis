import re
import os
import csv
import six
import sys
import ast
import unidecode
#import sqlite3
import string
import configparser
import json

import pandas as pd
import numpy as np

from tweepy import OAuthHandler, API
#from google.cloud import language
#from google.api_core.exceptions import InvalidArgument

from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

from tweepy import Stream
from tweepy.streaming import StreamListener

from collections import Counter
from pymongo import MongoClient, errors

from wordcloud import WordCloud
from PIL import Image

# Extraemos las claves para la API de twitter del environment

# [twitter keys]
CONSUMER_KEY = os.environ['CONSUMER_KEY']
CONSUMER_SECRET = os.environ['CONSUMER_SECRET']
CONSUMER_TOKEN = os.environ['CONSUMER_TOKEN']
CONSUMER_TOKEN_SECRET = os.environ['CONSUMER_TOKEN_SECRET']

## leemos el archivo de configuracion con las rutas de los ficheros
config = configparser.ConfigParser()
_=config.read("twitter.conf")
## Paths
OUTFILE_PATH = ast.literal_eval(config.get("paths", "OUTFILE_PATH"))
ALERTWORDS_PATH = ast.literal_eval(config.get("paths", "ALERTWORDS_PATH"))

SENTIMENT = ast.literal_eval(config.get("vars", "SENTIMENT"))
#TRACK='@metro_madrid'
TRACK = ast.literal_eval(config.get("vars", "TRACK"))

### rutas donde estan los ficheros para el lexicon
POS_FILE = ast.literal_eval(config.get("paths", "POS_FILE"))
EMOJI_POS = ast.literal_eval(config.get("paths", "EMOJI_POS"))
NEG_FILE = ast.literal_eval(config.get("paths", "NEG_FILE"))
EMOJI_NEG = ast.literal_eval(config.get("paths", "EMOJI_NEG"))

cols=ast.literal_eval(config.get("vars","cols"))

## Ruta de la MongoDB [MongoDBAtlas]
MONGODBATLAS = os.environ['URI']

### TEXT ANALYTICS ###
######################



class TextAnalysis(object):
    """
    TextAnalysis class: functions for text analytics, word counts, word clouds, etc.
    """

    def __init__(self, text, remove_stopwords=True, remove_accents=True, stemmer = SnowballStemmer('spanish')):
        """
        Analize text data:
        :param text: text
        :param remove_stopwords: True to remove stopwords
        :param stemmer: SnowballStemmer(language)
        :type text: str
        :type remove_stopwords: bool
        :type stemmer: SnowballStemmer
        """

        self.text = text
        self.stemmer = stemmer

        if remove_stopwords:
            self.stop = self.stop_words() # list of stop-words. new ones can be added as argument
        else:
            self.stop = []

        if remove_accents:
            self.text=self.remove_accents(self.text)

        self.tokens = self.tokenize_words()
        self.tokens = self.remove_stop_words(self.tokens)
        self.words_stemmed = self.stem_words(self.tokens)

    def stop_words(self, additional=[]):
        """
        inizialize stop words that are going to be removed in the text.
        :param additional: additional stop words
        :return: stopwords in spanish
        """
        punctuation = list(string.punctuation)
        stop = stopwords.words('spanish') + punctuation + ['rt', 'via', 'saludo', '...'] + additional
        return stop

    def tokenize_words(self):
        """
        convert words to tokens
        :return: words tokenized
        """
        self.tokens = word_tokenize(self.text)
        return self.tokens

    def remove_stop_words(self, tokens):
        """
        remove stop words from tokens
        :return: tokens without stopwords.
        """
        wout=[]
        for word in tokens:
            word = word.lower()
            if not word in self.stop:
                wout.append(word)
        return wout

    def word_counter(self):
        """
        Count the number of words in text
        :return: dictionary with counts of each word.
        """
        counter = Counter()
        for i in self.tokens:
                counter[i]+= 1
        return counter

    def remove_accents(self, string):
        """
        remove the accents of the text
        :param string: string
        :return: return a string without accents.
        """
        return unidecode.unidecode(string)

    def worcloud_from_dict(self, dictionary, fout_name='../out/wc.png', masked=True, **kargs):
        if masked:
            image_name='../data/front_train_bgw_2.png'
            train_mask = np.array(Image.open(image_name))
            wc = WordCloud(background_color="white", max_words=2000, mask=train_mask,
                 colormap='magma',width=1600, height=800,)
        else:
            wc = WordCloud(**kargs)
        wc.generate_from_frequencies(dictionary)
        wc.to_file(fout_name)

    def create_wordcloud(self, fout_name='../out/wordcloud.png', masked=True, **kargs):
        """
        create a wordcloud from text
        :param fout_name: name of the output figure
        :param **kargs: arguments to pass to the WordCloud Class.
        """
        counter = self.word_counter()
        self.worcloud_from_dict(counter,fout_name=fout_name,masked=masked)

    def most_freq_words(self, col='hashtags'):
        """
        return the most frequence words of a col in the tweets: by default, hashtags:
        :return: most common words in text
        """
        wc = self.word_counter().most_common()
        return {i:j for i,j in wc}

    def get_sentiment_analysis(self, method='lexicon'):
        """
        returns the sentiment analysis of the text.
        :param method: posible lexicon or api (google cloud api)
        :return: the score and the sentiment of the text.
        """
        if not hasattr(self, 'sentiment_lexicon'):
            self.sentiment_lexicon = SentimentAnalysis()
        return self.sentiment_lexicon.get_sentiment_analysis(self.text, method=method)

    def stem_words(self, words):
        """
        stem words:
        :param words: list of words
        :return: words stemmed

        """
        return [self.stemmer.stem(i) for i in  words]

    def mask_words_in_tokens(self, words, stemmed = True):
        if stemmed:
            words_stemmed = self.stem_words(words)
            return [word in self.words_stemmed for word in words_stemmed]
        else:
            return [word in self.tokens for word in words]

    def words_in_tokens(self, words, stemmed = True):
        if type(words) == str:
            words=[words]
        mask = self.mask_words_in_tokens(words, stemmed = stemmed)
        return [words[i] for i in range(len(words)) if mask[i]]

    def check_if_words_in_tokens(self,words,stemmed=True):
        """
        check if  words is in the token list
        :param words: list of words
        :return: bool
        """
        if type(words) == str:
            words = [words]
        return any(self.mask_words_in_tokens(words, stemmed = stemmed))


class SentimentAnalysis(object):
    """
    SentimentAnalysis class: functions that calculate sentiment of entering text.
    """

    def __init__(self):
        self.tokens_pos = self.get_tokens_from_file(POS_FILE)
        self.emoji_pos = self.get_tokens_from_file(EMOJI_POS)
        self.tokens_neg = self.get_tokens_from_file(NEG_FILE)
        self.emoji_neg = self.get_tokens_from_file(EMOJI_NEG)

    def check_sentiment_of_emoji(self, emoji):
        """
        check if an emoji has a positive or negative sentiment
        """
        if emoji in self.emoji_pos:
            return 1
        elif emoji in self.emoji_neg:
            return -1
        else:
             return 0

    def check_sentiment_of_word(self, word):
        """
        check if a word has a positive or negative sentiment.
        """
        if word in self.tokens_pos:
            return 1
        elif word in self.tokens_neg:
            return -1
        else:
            return 0

    def check_sentiment_is_positive(self, word):
        if ((word in self.tokens_pos) | (word in self.emoji_pos)):
            return True
        return False

    def check_sentiment_is_negative(self, word):
        if ((word in self.tokens_neg) | (word in self.emoji_neg)):
            return True
        return False

    def check_sentiment_of_text(self, text):
        """
        balance a text with the total sum of sentiments
        """
        tokens = word_tokenize(text.lower())
        positive=0
        negative=0
        for token in tokens:
            positive+= self.check_sentiment_is_positive(token)
            negative+= self.check_sentiment_is_negative(token)
        magnitude=positive-negative
        if ((magnitude)!=0):
            score=np.divide(positive+negative,positive-negative)
        else:
            score=0

        return score,magnitude

    def get_sentiment_analysis(self, text, method='lexicon'):
        if method == 'lexicon':
            return self.check_sentiment_of_text(text)
        elif method == 'api':
            return self.get_sentiment_analysis_api(text)
        elif not method:
            return None,None
        else:
            raise ValueError("elegir metodo lexicon o api")

    def get_tokens_from_file(self, file_name):
        """
        Reads the lexicon and returns the set containing all the tokens.
        The tokens will include words / emoticons / emojis.

        Args:
            lexiconName (str): Path to the lexicon

        Returns:
            tokens (set): set containing all the tokens
        """
        with open(file_name, encoding="latin-1") as f:
            tokens = set()
            for line in f.readlines():
                line = line.strip()
                tokens.add(line)
        return tokens

    def get_sentiment_analysis_api(self, text, language_client=None):
        """
        Analysis de sentimiento a partir de la API de GCP.
        :return: score y sentimiento.
        """

        if not language_client:
            language_client = language.LanguageServiceClient()

        if isinstance(text, six.binary_type):
            text = text.decode('utf-8')

        document = language.types.Document(
            content = text.encode('utf-8'),
            type = language.enums.Document.Type.PLAIN_TEXT)

        encoding = language.enums.EncodingType.UTF32
        if sys.maxunicode == 65535:
            encoding = language.enums.EncodingType.UTF16

        # Analyze the sentiment
        try:
            annotations = language_client.analyze_sentiment(document, encoding).document_sentiment
            return annotations.score, annotations.magnitude
        except InvalidArgument:
            return None,None


class AlertWords(TextAnalysis):

    def __init__(self, alertwords_file, stemmer = SnowballStemmer('spanish')):
        """
        Important words that are interesting in the streaming
        """
        self.__file = alertwords_file
        self.alert_words = self.load_alert_words(alertwords_file)
        TextAnalysis.__init__(self, self.text, remove_stopwords=False)

    def __repr__(self):
        return "{0} alert words from {1}".format(len(self.tokens), self.__file)

    def load_alert_words(self, alertwords_file):
        """
        load the words that are going to be used as a warning
        """
        with open(alertwords_file) as f:
            self.text = f.read()

    def get_stemmed_alert_words(self):
        """
        stem alert words to get all words that starts with the same lexema.
        """
        return self.words_stemmed




### Analysis of tweets ###
##########################




class TweetData(TextAnalysis):

    def __init__(self, tweet, tweet_mode='extended', sentiment='lexicon',
                 remove_stopwords=True, alertwords=True, fname=ALERTWORDS_PATH):
        """
        Analyze and preprocess tweet data
        """
        self.tweet = tweet
        self.tweet_mode = tweet_mode
        self.sentiment = sentiment
        self.text = self.text_of_tweet()
        if alertwords:
            self.alertwords = AlertWords(fname)
        else:
            self.alertwords = None
        TextAnalysis.__init__(self, self.text, remove_stopwords = remove_stopwords)

    def __repr__(self):
        return "tweet {0} from {1}".format(self.tweet.id, self.tweet.author.id)

    def text_of_tweet(self):
        """
        Obtain the text of a tweetpy.tweet class
        :return: text of the tweet.
        """
        try:
            return self.tweet.full_text
        except AttributeError:
            return self.tweet.text

        if self.tweet_mode=="extended":
            method="full_text"
        else:
            method="text"
        return getattr(self.tweet, method)

    def get_tweet_entities(self, sentiment='lexicon'):
        """
        extract the tweet entities (citation, hashtags, urls, media and symbols)
        :return: dictionary with tweet entities
        """
        new_text, citations, hashtags, urls, media, symbols = self.tweets_split_rts_hstgs_text()
        if sentiment:
            sent_score,sent_mag = self.get_sentiment_analysis(method=sentiment)
        else:
            sent_score,sent_mag = [None,None]
        if self.alertwords:
            alert = self.check_if_text_has_wordalerts()
        else:
            alert=None

        data={'text':new_text,
              'citations':citations,
              'hashtags':hashtags,
              'urls':urls,
              'media':media,
              'symbols':symbols,
              'sent_score':sent_score,
              'sent_mag':sent_mag,
              'alert':alert,
              'sent_method':SENTIMENT,
              'track':TRACK}
        return data

    def metadata_tweet(self):
        """
        extract metadata from the tweet
        :return: dictionary with the metadata
        """
        data = {"author_id": self.tweet.author.id_str,
                "tweet_id": self.tweet.id_str,
                "created_at": self.tweet.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "favorite_count": self.tweet.favorite_count,
                "retweet_count": self.tweet.retweet_count}
        return data

    def preprocess_tweet(self, sentiment='lexicon'):
        """
        extract the data and metadata from tweet.
        It is possible to do sentiment analysis of text as well.
        :return: dictionary with data of preprocessed tweet
        """
        data = self.get_tweet_entities(sentiment=sentiment)
        metadata = self.metadata_tweet()
        data.update(metadata)
        return data

    def data_to_json(self, dc):
        return json.dumps(dc, ensure_ascii=False)

    def preprocess_tweet_json(self, sentiment='lexicon'):
        return self.data_to_json(self.preprocess_tweet(method=sentiment))

    def tweets_split_rts_hstgs_text(self):
        """
        splits hashtags and other info of the text
        """
        entities = self.tweet.entities
        hashtags = ['#'+hs['text'] for hs in entities["hashtags"]]
        citations = ['@'+us_m['screen_name'] for us_m in entities["user_mentions"]]
        urls = [url['url'] for url in  entities["urls"]]
        symbols = [symbol for symbol in  entities["symbols"]]
        #symbols = [ in self.text_of_tweet()]
        retweet = None
        if hasattr(self.tweet,'retweeted_status'):
            retweet = self.tweet.retweeted_status.id_str
        media = []
        if 'media' in entities:
            media=[med['url'] for med in entities['media']]

        data = list(filter(None,['RT|:', '|'.join(hashtags), '|'.join(citations), '|'.join(urls), '|'.join(symbols), '|'.join(media)]))
        if data:
            regex = '|'.join(data)
        else:
            regex = ''
        new_text = re.sub(regex,'',self.text_of_tweet()).strip()

        return new_text, citations, hashtags, urls, media, symbols

    def check_if_text_has_wordalerts(self):
        return self.check_if_words_in_tokens(self.alertwords.words_stemmed)


class AllTweetsAnalytics(object):
    """
    Class AllTweetsAnalytics
    """
    def __init__(self, tweets=None, mode='historical', alertwords=True, preprocess=False, sentiment='lexicon'):
        self.twitter_auth(mode=mode)
        self.tweets = tweets
        if preprocess:
            self.tweets=self.preprocess_tweets(self.tweets, sentiment=sentiment)
        else:
            self.processed_tweets = False
        if mode:
            self.twitter_auth(mode=mode)
        else:
            self.api=None

    def twitter_auth(self, mode='historical'):
        self.api = TwitterAuthentification(mode = mode).auth

    def get_attribute_from_tweets(self, col, as_string=True):
        values = []
        data = [tweet[col] for tweet in self.tweets if tweet[col]]
        if col == 'text':
            values = data
        else:
            _=list(map(values.extend, data))
        if as_string:
            return ', '.join(values)
        else:
            return values

    def get_tweets(self, screen_name, count = 200, tweet_mode = 'extended', max_id = None):
        if not self.api: raise ValueError("connect before with twitter_auth method")
        return self.api.search(q = screen_name, count = count, tweet_mode = tweet_mode, max_id = max_id)

    def preprocess_tweets(self, alltweets, sentiment='lexicon'):
        if self.tweets:
            alltweets = [TweetData(tweet).preprocess_tweet(sentiment=sentiment) for tweet in alltweets]
            #self.tweets = alltweets
        else:
            raise ValueError("tweets not defined yet")
        return alltweets

    def get_historical_tweets(self, screen_name, count = 200, tweet_mode = 'extended',
                              max_count = 1e9, preprocess_tweet = True, sentiment='lexicon'):
        """
        Extraccion de todos los tweets
        """
        #Twitter only allows access to a most recent 3240 tweets with this method

        self.processed_tweets = preprocess_tweet
        #initialize a list to hold all the tweepy Tweets
        alltweets = []
        oldest = None
        while True:
            #make initial request for most recent tweets (200 is the maximum allowed count)
            new_tweets = self.get_tweets(screen_name, count = count,
                                         tweet_mode = tweet_mode, max_id = oldest)
            #api.search(q=screen_name,count=count,tweet_mode=tweet_mode,max_id=oldest)

            if len(new_tweets)<=0: break
            #save most recent tweets
            alltweets.extend(new_tweets)

            print ("...%s tweets downloaded so far" % (len(alltweets)))
            #save the id of the oldest tweet less one
            oldest = alltweets[-1].id - 1

            if len(alltweets) >= max_count:
                break
        self.tweets=alltweets
        if preprocess_tweet:
            self.tweets=self.preprocess_tweets(alltweets,sentiment=sentiment)
            self.processed_tweets = True
        else:
            self.tweets = [TweetData(tweet) for tweet in alltweets]

    def check_if_tweets_not_empty(self):
        if self.tweets:
            return True
        else:
            return False

    def most_freq_words(self, col):
        try:
            data = TextAnalysis(self.get_attribute_from_tweets(col))
            return data.most_freq_words()
        except AttributeError:
            raise ValueError("not preprocessed tweets")

    def most_freq_citations(self):
        return self.most_freq_words(col = 'citations')

    def most_freq_hastags(self):
        return self.most_freq_words(col = 'hashtags')

    def create_wordcloud(self,col='text',**kargs):
        data=TextAnalysis(self.get_attribute_from_tweets(col))
        data.tokens = data.tokenize_words()
        data.tokens = data.remove_stop_words(data.tokens)
        data.create_wordcloud(**kargs)

    def create_wordcloud_tweets_with_alertwords(self,col='text',**kargs):
        alerts = AllTweetsAnalytics(tweets=self.preprocess_tweets([tweet.tweet for tweet in self.check_if_tweet_has_wordalerts()]))
        alerts.create_wordcloud(col=col,fout_name='../out/wordcloud_aw.png',**kargs)

    def check_if_tweet_has_wordalerts(self):
        if self.processed_tweets:
            raise ValueError('cannot do it with preprocessed tweets')
        return [tweet for tweet in self.tweets if tweet.check_if_text_has_wordalerts()]

    def return_tweets_separated_by_sentiment(self):
        if not self.processed_tweets:
            raise ValueError('cannot do it with preprocessed tweets')
        positive=AllTweetsAnalytics(tweets=[tweet for tweet in self.tweets if tweet['sent_score']>0])
        negative=AllTweetsAnalytics(tweets=[tweet for tweet in self.tweets if tweet['sent_score']<0])
        return positive,negative




###### TWITTER API ##########
#############################




class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    def __init__(self,):
        StreamListener.__init__(self)

    def on_status(self, status):
        """
        print text of the tweet
        """
        tweet = TweetData(status).preprocess_tweet(sentiment=SENTIMENT)
        print('Tweet : ' + tweet)
        return True

    def on_error(self, status):
        print('Error with status: ' + str(status))
        return True

    def on_timeout(self):
        print('Timeout...')
        return True # To continue listening


class TwitterAuthentification(object):
    """
    Twitter authentification class: manages how we connect to twitter
    """
    def __init__(self, mode='historical', listener=StdOutListener()):
        if mode == 'historical':
            self.auth = self.auth_twitter_api()
        elif mode == 'stream':
            self.auth = self.auth_twitter_stream(listener=listener)
        else:
            raise ValueError("allowed: mode: historical or stream")

    def auth_twitter(self):
        """
        authorization twitter
        :param CONSUMER_KEY: twitter credentials
        :param CONSUMER_SECRET: twitter credentials
        """
        auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(CONSUMER_TOKEN, CONSUMER_TOKEN_SECRET)
        return auth

    def auth_twitter_api(self):
        """
        authorize twitter, initialize API for download historial data
        """
        print("tweepy.API call ...")
        return API(self.auth_twitter())

    def auth_twitter_stream(self, listener=StdOutListener(), **kargs):
        """
        authorize twitter, initialize Stream for stream data
        """
        return Stream(self.auth_twitter(), listener, **kargs)




### DATA STORING ####
#####################




class MongoDBDatabase(object):

    def __init__(self, client_name ='tweets_analytics', collection='tweets'):
        if "DB_PORT_27017_TCP_ADDR" in os.environ:
            client = MongoClient(os.environ['DB_PORT_27017_TCP_ADDR'],27017)
        else:
            client = MongoClient()
        db = client[client_name]
        self.coll = db[collection]

    def insert_doc_in_mongo_db(self,doc):
        try:
            return self.coll.insert_one(doc).inserted_id
        except errors.DuplicateKeyError:
            pass
            #print ("doc already exists")

    def search_db(self,query=None):
        return self.coll.find(query)

    def check_if_exists(self,id_n):
        return self.search_db({"_id": id_n}).count() >0


class TweetsMongoDBD(MongoDBDatabase):
    def __init__(self, client_name='tweets_analytics', collection='tweets'):
        MongoDBDatabase.__init__(self,client_name = client_name , collection = collection)

    def get_all_attr(self,attribute = "hashtags",limit=0,preprocess=False):
        query = {attribute:{"$ne":[]}}
        tweets = AllTweetsAnalytics(tweets = list(self.search_db(query).limit(limit)), preprocess=preprocess)
        return tweets

    def get_text(self, preprocess=False):
        return self.get_all_attr("text", preprocess=preprocess)

    def get_hashtags(self):
        return self.get_all_attr(attribute="hashtags", limit = 25).most_freq_words(col="hashtags")

    def get_citations(self):
        return self.get_all_attr(attribute="citations", limit = 25).most_freq_words(col="citations")

    def insert_tweet(self, tweet):
        if type(tweet)==dict:
            tweet['_id'] = tweet.pop('tweet_id')
            if self.check_if_exists(tweet["_id"]):
                return False
            _ = self.insert_doc_in_mongo_db(tweet)
            return True
        else:
            raise TypeError("introduce preprocessed tweet")

    def insert_in_mongodb_multiple_tweets(self, alltweets):
        """
        Insert alltweets in mongodb
        :param alltweets: list of alltweets
        """
        if type(alltweets) == dict:
            alltweets = alltweets.values()
        for tweet in alltweets:
            _=self.insert_tweet(tweet)


class Sqlite3Database(object):

    def __init__(self, db_name="../twit_data.db", table_name='tweets_analytics'):
        self.db_name = db_name
        self.table_name = table_name
        self.conn = sqlite3.connect(db_name)
        c = conn.cursor()
        cols_type={col:'TEXT' for col in cols}
        fields = ', '.join(['{0} {1}'.format(i, j) for i, j in cols_type.items()])
        cmd = "CREATE TABLE {0} ({1})".format(table_name, fields)
        c.execute(cmd)
        conn.commit()
        #conn.close()

    def save_to_db_sqlite3(self, alltweets):
        if type(alltweets) != list:
            alltweets = [alltweets]
        c = self.conn.cursor()
        tweets_data = [[tweet[col] for col in cols] for tweet in alltweets]
        cmd = 'insert into {0} values ({1})'.format(self.table_name,','.join(len(cols)*['?']))
        c.executemany(cmd,tweets_data)
        c.commit()


class SavetoFileTweets(object):

    def __init__(self, fname, alltweets):
        """
        Save multiple tweets into a file.
        :param fname: name of the output file
        :param alltweets: lists of tweets that are going to be saved.
        """
        self.fname = fname
        self.alltweets = alltweets
        self.save_tweets_in_file()

    def save_tweets_in_file(self):
        """
        save all tweets in a file
        """
        ids=self.check_if_file_exist()
        with open(self.fname, 'w') as f:
            writer = csv.writer(f)
            #language_client = language.Client()

            writer.writerow(cols)
            for tw in self.alltweets:
                if self.check_if_tweet_exists(tw['tweet_id'], ids):
                    continue
                writer.writerow([str(tw[col]) for col in cols])

    def check_if_file_exist(self):
        """
        if file already exists returns the ids of the tweets to not have duplicates
        """
        if os.path.exists(self.fname):
            ids = pd.read_csv(self.fname, usecols = ['tweet_id'], index_col = ['tweet_id']).index.unique()
        else:
            ids = []
        return ids

    def check_if_tweet_exists(self,tweet_id,ids):
        """
        check if tweet is already in file
        :param tweet_id: identifier of tweet
        :param ids: list of identifiers
        :return bool: true or false
        """
        if tweet_id in ids:
            return True
        else:
            return False
