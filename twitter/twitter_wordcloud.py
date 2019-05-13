from twitter import TweetsMongoDBD, TextAnalysis,AllTweetsAnalytics

### attributo con el que se va a general el wordcloud. Posibles: text, hashtags y citations
attribute="citations"
fout_name="../out/wordcloud_{0}.png".format(attribute)

"""
generacion de un wordcloud a partir de los tweets que hay en la base de datos
"""

if __name__ == '__main__':
    mb=TweetsMongoDBD()
    if attribute=="text":
        _=mb.get_text().create_wordcloud(fout_name=fout_name)
    elif attribute=='hashtags':
        TextAnalysis('').worcloud_from_dict(mb.get_hashtags(),fout_name=fout_name, masked=True)
    elif attribute=='citations':
        TextAnalysis('').worcloud_from_dict(mb.get_citations(),fout_name=fout_name, masked=True)
    elif attribute=='sentiment':
        ### positivos
        fout_name="out/wordcloud_{0}pos.png".format(attribute)
        tw=AllTweetsAnalytics(list(mb.search_db({"sent_score":{"$gt":0}})))
        tw.create_wordcloud(fout_name=fout_name)
        ### negativos
        fout_name="out/wordcloud_{0}neg.png".format(attribute)
        tw=AllTweetsAnalytics(list(mb.search_db({"sent_score":{"$lt":0}})))
        tw.create_wordcloud(fout_name=fout_name)

    else:
        raise ValuError("only valid text, hashtags, citations and sentiment as attribute value")
    print ("Done")
