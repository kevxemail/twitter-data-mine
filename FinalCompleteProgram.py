import nltk.corpus
from nltk.corpus import twitter_samples
import nltk
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.corpus import stopwords

stop_words = stopwords.words("english")  # Get the english stop words only
from nltk import FreqDist
import random
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections

import tweepy as tw
import warnings

warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")  # A LOT OF IMPORTS
# Don't want people seeing this lol
consumer_key =  # API Key
consumer_secret =  # API Key secret
bearer_token = 
access_token = 
access_token_secret = 

auth = tw.OAuthHandler(
    consumer_key, consumer_secret
)  # All to connect Twitter with my code
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


def remove_noise(tokens):
    lemmatizer = (
        WordNetLemmatizer()
    )  # Create a WordNetLemmatizer object https://www.nltk.org/_modules/nltk/stem/wordnet.html, understand wn.morphy(word, pos)
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        word = word.lower()
        if (
            "http" in word
            or "https" in word
            or ".com" in word
            or ".edu" in word
            or ".org" in word
            or ".net" in word
            or ".biz" in word
            or ".edu" in word
            or ".gov" in word
            or ".mil" in word
            or ".in" in word
            or ".br" in word
            or ".it" in word
            or ".ca" in word
            or ".mx" in word
            or ".uk" in word
            or ".tw" in word
            or ".cn" in word
            or word[0] == "@"
        ):
            word = ""  # Replace links with nothing
        if tag.startswith("NN"):
            pos = "n"  # For WordNetLemmatizer/wn, this represents a noun
        elif tag.startswith("VB"):
            pos = "v"  # Represents a verb
        else:
            pos = "a"  # Represents an adjective
        if (
            len(word) > 0 and word not in string.punctuation and word not in stop_words
        ):  # Filters out the removed links, punctuation, and stop words
            lemmatized_sentence.append(
                lemmatizer.lemmatize(word, pos)
            )  # "lemmatize" the word, turning it into its root word
        # "pos" variable lets the morphy method know which type to look for in the database, because there are different suffixes and endings attached to each which could be converted. https://wordnet.princeton.edu/documentation/morphy7wn
    return lemmatized_sentence


def get_tweets_for_model(cleaned_tokens):
    temporary_list = (
        []
    )  # List to hold dictionaries for each tweet, where it becomes token:True for each word in the tweet
    for tokens in cleaned_tokens:
        temporary = {}  # The temporary dictionaries that will be added to the list
        for token in tokens:
            temporary[
                token
            ] = True  # token:True, keep adding dictionary values to the temp dictionary for this one tweet
        temporary_list.append(
            temporary
        )  # Add the entire dictionary of {token:True} for that particular tweet to the list
    return temporary_list
    # for tweet_tokens in cleaned_tokens:
    # yield dict([token, True] for token in tweet_tokens)


def get_all_words(
    cleaned_tokens,
):  # Meant to add all the tokens in cleaned_tokens to a new list, so it basically flattens the list, turning the list of lists of strings into just one huge list of strings
    for tokens in cleaned_tokens:
        for token in tokens:
            yield token  # It'll add all the tokens into the list https://www.geeksforgeeks.org/use-yield-keyword-instead-return-keyword-python/


# yield sends back the value to the caller, but then continues to run the body of the loop to send back a sequence of values
# with return you'd just append it all to a temp list and return temp
# yield is used with python generators, if it uses yield the funciton becomes a generator function


def twitter_data_analyze():
    query = str(input("Enter a search query: "))
    query2 = query + " -filter:retweets"  # Filter out the retweets
    tweets = tw.Cursor(api.search_tweets, q=query2, lang="en").items(
        1000
    )  # Get tweets, search_tweets only goes back 7 days
    all_tweets = [
        remove_noise(word_tokenize(tweet.text)) for tweet in tweets
    ]  # Create a tokenized, cleaned list of lists of strings, each list inside the list is a tweet

    count_pos = 0
    count_neg = 0
    for tweet in all_tweets:
        formatted_custom_tokens = dict(
            [token, True] for token in tweet
        )  # Basically a nested for loop, outer loop goes through each tweet, inner loop goes through each token
        if classifier.classify(formatted_custom_tokens) == "Positive":
            count_pos += 1
        else:
            count_neg += 1
    clean_tweets_no_urls = pd.DataFrame(
        [("Positive", count_pos), ("Negative", count_neg)],
        columns=["Sentiment", "Count"],
    )  # Make a Panda dataframe for the count_pos and count_neg
    # Create a horizontal bar graph with the sentiments
    fig, ax = plt.subplots(
        figsize=(8, 8)
    )  # Initialize the plot and how big it will be using matplotlib

    # Plot horizontal bar graph
    """
    Plot horizontal bar graph
    ".sort_values(by='count')" puts the highest count values at the top
    "ax=ax" in the plot.barh thing essentially says to use those 
    Honestly not sure how the matplotlib interacts with the panda, research more later
    """
    clean_tweets_no_urls.plot.barh(x="Sentiment", y="Count", ax=ax, color="purple")

    ax.set_title("Sentiment of the search query: " + query)

    plt.show()


def displayfrequency(cleaned_pos, cleaned_neg):
    all_pos_words = get_all_words(cleaned_pos)
    all_neg_words = get_all_words(cleaned_neg)

    freq_pos = FreqDist(
        all_pos_words
    )  # Use FreqDist class to find out the which words are most common
    freq_neg = FreqDist(all_neg_words)


"""
Training model start
"""
pos_tweets = twitter_samples.strings(
    "positive_tweets.json"
)  # strings() method of twitter_samples will print all of the tweets within a dataset as strings
neg_tweets = twitter_samples.strings("negative_tweets.json")
twentyk_tweets = twitter_samples.strings("tweets.20150430-223406.json")
tokenized_pos = twitter_samples.tokenized("positive_tweets.json")
tokenized_neg = twitter_samples.tokenized("negative_tweets.json")

cleaned_pos = [
    remove_noise(cleaned) for cleaned in tokenized_pos
]  # Use the remove_noise method to clean each list of strings stored
cleaned_neg = [remove_noise(cleaned) for cleaned in tokenized_neg]

pos_tokens_dict = get_tweets_for_model(
    cleaned_pos
)  # Create lists of dictionaries with the form {token:value} for each tweet using the method above
neg_tokens_dict = get_tweets_for_model(cleaned_neg)
pos_dataset = [
    (tweets, "Positive") for tweets in pos_tokens_dict
]  # Tags each dictionary value with positive or negative, creates a list of tuples
neg_dataset = [(tweets, "Negative") for tweets in neg_tokens_dict]

data = pos_dataset + neg_dataset  # Put the datasets into one list
random.shuffle(
    data
)  # Use random to shuffle the list's items around, so it isn't just first half positive second half negative
train_data = data[:7000]  # First 7000 tweets are used to train the model
test_data = data[7000:]  # Last 3000 tweets are used to verify the accuracy of the model

classifier = nltk.NaiveBayesClassifier.train(
    train_data
)  # This one method trains the naivebayes classifier, uses the "NaiveBayesClassifier" import
print(
    "Accuracy is:", classify.accuracy(classifier, test_data) * 100
)  # This method checks the accuracy with the test data, uses the "classify" import
print(classifier.show_most_informative_features(10))

"""
Training model end
"""

twitter_data_analyze()
