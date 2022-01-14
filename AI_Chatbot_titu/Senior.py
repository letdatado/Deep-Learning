# -*- coding: utf-8 -*-
"""
Contains the function for SeniorBot
"""

# importing the libraries
import re
import nltk
import string
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import aiml


def seniorbot(topic):
    '''
    This function serve as the brain behind 'Senior bot'

    '''
    
    # LOAD TOPIC FROM WIKI API
    page_object = wikipedia.page(str(topic))
    my_article = page_object.content

    # MAKE THE CORPUS
    corpus = '' # one big string !! 
    

    for words in my_article: # This for loop loops for every word present in the corpus
        corpus += words
    
    # CLEAN THE CORPUS
    corpus = corpus.lower()     # we lower case it here (we could do it upper case too)


    corpus = re.sub(r'\[[0-9]*\]', ' ', corpus) 
    corpus = re.sub(r'\s+', ' ', corpus)
    corpus = re.sub(r'\=+', '', corpus)
    
    
    # TOKENIZE
    sent_tokens = nltk.sent_tokenize(corpus)
    word_tokens = nltk.word_tokenize(corpus)

    # LEMMATIZE
    lemma = nltk.stem.WordNetLemmatizer()

    def lemtokens(tokens):
        return [lemma.lemmatize(token) for token in tokens]
    dict_remove_punctuations = dict((ord(punct), None) for punct in string.punctuation)

    def lemnormalize(text):
        return lemtokens(nltk.word_tokenize(text.lower().translate(dict_remove_punctuations)))


    # GENERATE THE RESPONSE
    def response(user_response):
        robot_response = ''
        TfidfVec = TfidfVectorizer(tokenizer=lemnormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1],tfidf)
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if(req_tfidf == 0):
            robot_response = robot_response + 'I am Sorry! I couldn\'t get it'
            return robot_response
        else:
            robot_response = robot_response + sent_tokens[idx]
            return robot_response 


    # PLAN THE CONVERSATION
    flag = True
    print('Senior Bot: Lets try to get to know about {} in detail, If you want to exit anytime, just type \'bye\''.format(topic))
    while(flag == True):
        user_response = input()
        user_response = user_response.lower()
        if(user_response != 'bye'):
            if(user_response == 'thanks' or user_response == 'thank you'):
                flag = False
                print('Senior Bot: You are welcome')
            else:

                sent_tokens.append(user_response)
                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print('Bot: ', end='')
                print(response(user_response))
                sent_tokens.remove(user_response)
        else:
            flag = False
            print('Senior Bot: Goodbye! Take care :))')
            
            
      