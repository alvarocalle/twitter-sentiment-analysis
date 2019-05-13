#from __future__ import absolute_import, print_function
from twitter import AllTweetsAnalytics
from twitter import SavetoFileTweets, MongoDBDatabase, TweetsMongoDBD

def main(screen_name, savehow='file', **kargs):
    """
    Descarga el historico de tweets para un determinado hashtag o usuario (screen_name).
    Es posible guardarlo como fichero o en la base de datos (savehow).
    :param screen_name: hashtag o usuario al que trackear.
    :param savehow: formato de como guardar un fichero.
    :param **kargs: argumentos opcionales que se le pasan a la funcion get_historical_tweets.
    """
    tweets=AllTweetsAnalytics()
    tweets.get_historical_tweets(screen_name,**kargs)
    alltweets=tweets.tweets
    #transform the tweepy tweets into a 2D array that will populate the csv
    #alltweets=[preprocess_tweet(tw) for tw in alltweets]
    if savehow=='file':
        fname='{0}/{1}_hist_tweets.csv'.format('out',screen_name[1::])
        SavetoFileTweets(fname,alltweets)
       # save_tweets_in_file(alltweets,fname)

    elif savehow == 'db':
        mb = TweetsMongoDBD()
        mb.insert_in_mongodb_multiple_tweets(alltweets)
    else:
        alltweets={tweet['tweet_id']: tweet for tweet in alltweets}
        print(data_to_json(alltweets))
    print("done")


if __name__ == '__main__':
    main('@metro_madrid', savehow='db', max_count=15000, preprocess_tweet=True)
    #main('@metro_madrid', savehow='file', max_count=100, preprocess_tweet=True)
