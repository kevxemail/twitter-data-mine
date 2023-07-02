import nltk

nltk.data.path.append(
    "~/git/kevinxiong1505525/git/twitter-data-mining/nltk_data"
)  # Set NLTK's path to this folder not my home directory
# ROUND 1 OF IMPORTS
import nltk.corpus

nltk.download("twitter_samples")
from nltk.corpus import twitter_samples

# ROUND 2 OF IMPORTS
nltk.download("punkt")

# ROUND 3 OF IMPORTS
nltk.download(
    "wordnet"
)  # Lexical database for English language that helps the script determine the base word
nltk.download(
    "averaged_perceptron_tagger"
)  # Required for the thing above, determines the context of a word in a sentence
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

# ROUND 4 OF IMPORTS
import string
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("wordnet")

# ROUND 5 OF IMPORTS
from nltk import FreqDist

# ROUND 6 OF IMPORTS
import random

# ROUND 7 OF IMPORTS
from nltk import classify
from nltk import NaiveBayesClassifier

# ROUND 8 OF IMPORTS
from nltk.tokenize import word_tokenize


"""
ROUND 1 OF IMPORTS - DOWNLOADING THE DATA AND STORING THEM WITH TWITTER_SAMPLES.STRINGS
Obtain the tweets from the json files
"""
pos_tweets = twitter_samples.strings(
    "positive_tweets.json"
)  # strings() method of twitter_samples will print all of the tweets within a dataset as strings
neg_tweets = twitter_samples.strings("negative_tweets.json")
twentyk_tweets = twitter_samples.strings("tweets.20150430-223406.json")

# print(len(pos_tweets)) # pos_tweets and neg_tweets are a list of strings!
# print(pos_tweets[0])

"""
ROUND 2 OF IMPORTS - TOKENIZING THE DATA WITH .TOKENIZED
.tokenized splits the list of strings into a list of lists with the strings divided by spaces
so [[word in tweet1, word in tweet1, word in tweet1, etc...], [word in tweet2, word in tweet2, word in tweet2, etc...], etc...]
Basically does the .split method on each node in one line of code

Uses punkt, which is a pre-trained model that helps tokenize words/sentences

Ex: It knows that a name may be written "A. Cai" and the presence of a period doesn't necesssarily end the sentence.
So .split but more advanced and conformed to natural language, takes a while because it's more complicated
"""
tokenized_pos = twitter_samples.tokenized("positive_tweets.json")
tokenized_neg = twitter_samples.tokenized("negative_tweets.json")

# print(type(tokenized_pos)) # .tokenized still keeps it as a list, but it splits the string in each node of the list into a list of strings divided by a ' '
# print(tokenized_pos[0])

"""
ROUND 3 OF IMPORTS (NORMALIZING THE DATA)
pos_tag was imported
Most common tags:
NNP: Noun, proper, singular
NN: Noun, common, singular or mass
IN: Preposition or conjunction, subordinating
VBG: Verb, gerund or present participle
VBN: Verb, past participle

In general, starting with NN = it's a noun, starting with VB = it's a verb
"""
# print(pos_tag(tokenized_pos[0])) # This one position of the total list, has a list of tuples with (tokenized string, tag)

"""
ROUND 4 - NORMALIZING THE DATA WITH WORDNET LEMMATIZER AND REMOVING "NOISE" FROM THE DATA USING IFELSE STATEMENTS, STOP_WORDS, AND STRING.PUNCTUATION
Need to remove "noise", like the words "is" "the" "a" add nothing
Removes links and @s
"""
stop_words = stopwords.words("english")  # Get the english stop words only


# collection_words = [], use this for collection words later
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


cleaned_pos = [
    remove_noise(cleaned) for cleaned in tokenized_pos
]  # Use the remove_noise method to clean each list of strings stored
cleaned_neg = [remove_noise(cleaned) for cleaned in tokenized_neg]

# print(tokenized_pos[500]) # See the huge difference the cleaned Tweet makes

"""
ROUND 5 OF IMPORTS - DETERMINING WORD DENSITY
Analyze word frequency distribution
"""


def get_all_words(
    cleaned_tokens,
):  # Meant to add all the tokens in cleaned_tokens to a new list, so it basically flattens the list, turning the list of lists of strings into just one huge list of strings
    for tokens in cleaned_tokens:
        for token in tokens:
            yield token  # It'll add all the tokens into the list https://www.geeksforgeeks.org/use-yield-keyword-instead-return-keyword-python/


# yield sends back the value to the caller, but then continues to run the body of the loop to send back a sequence of values
# with return you'd just append it all to a temp list and return temp
# yield is used with python generators, if it uses yield the funciton becomes a generator function
all_pos_words = get_all_words(cleaned_pos)
all_neg_words = get_all_words(cleaned_neg)

freq_pos = FreqDist(
    all_pos_words
)  # Use FreqDist class to find out the which words are most common
freq_neg = FreqDist(all_neg_words)

# print(freq_pos.most_common(10))

"""
ROUND 6 OF IMPORTS - PREPARING DATA FOR THE MODEL
Prepare data for Naive Bayes, Naive Bayes needs the data in dictionary form where the token is the key, and the value is "True"
A classifier based on the Naive Bayes algorithm.  In order to find the
probability for a label, this algorithm first uses the Bayes rule to
express P(label|features) in terms of P(label) and P(features|label):

|                       P(label) * P(features|label)
|  P(label|features) = ------------------------------
|                              P(features)

The algorithm then makes the 'naive' assumption that all features are
independent, given the label:

|                       P(label) * P(f1|label) * ... * P(fn|label)
|  P(label|features) = --------------------------------------------
|                                         P(features)

Rather than computing P(features) explicitly, the algorithm just
calculates the numerator for each label, and normalizes them so they
sum to one:

|                       P(label) * P(f1|label) * ... * P(fn|label)
|  P(label|features) = --------------------------------------------
|                        SUM[l]( P(l) * P(f1|l) * ... * P(fn|l) )

CHECK THE SLIDES YOU DID FOR NAIVE BAYES AND EMAIL CLASSIFICATION
"""


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


pos_tokens_dict = get_tweets_for_model(
    cleaned_pos
)  # Create lists of dictionaries with the form {token:value} for each tweet using the method above
neg_tokens_dict = get_tweets_for_model(cleaned_neg)

pos_dataset = [
    (tweets, "Positive") for tweets in pos_tokens_dict
]  # Tags each dictionary value with positive or negative, creates a list of tuples
neg_dataset = [(tweets, "Negative") for tweets in neg_tokens_dict]

# print(pos_dataset[0])

# Can't just use (tweets, "Positive"), have to create individual dictionaries
# for each node or else NaiveBayesClassifier won't work since it requires (dict{string : boolean, numbers, or string}, tag)

data = pos_dataset + neg_dataset  # Put the datasets into one list

random.shuffle(
    data
)  # Use random to shuffle the list's items around, so it isn't just first half positive second half negative

train_data = data[:7000]  # First 7000 tweets are used to train the model
# print(train_data)
test_data = data[7000:]  # Last 3000 tweets are used to verify the accuracy of the model

"""
ROUND 7 OF IMPORTS - TRAIN THE MODEL AND TEST FOR ACCURACY
"""
classifier = nltk.NaiveBayesClassifier.train(
    train_data
)  # This one method trains the naivebayes classifier, uses the "NaiveBayesClassifier" import
print(
    "Accuracy is:", classify.accuracy(classifier, test_data) * 100
)  # This method checks the accuracy with the test data, uses the "classify" import

print(classifier.show_most_informative_features(10))

"""
ROUND 8 OF IMPORTS - TEST YOUR OWN TWEETS
"""

custom_tweet = (
    "I ordered just once from TerribleCo, they screwed up, never used the app again."
)
custom_tokens = remove_noise(
    word_tokenize(custom_tweet)
)  # word_tokenize from the import "from nltk.tokenize import word_tokenize" tokenizes it, as discussed befoer with the .tokenized from the punkt model

formatted_custom_tokens = dict(
    [token, True] for token in custom_tokens
)  # Comprehension that turns it into the properly formatted dict[token:True] format

print(
    classifier.classify(formatted_custom_tokens)
)  # Let's see if this one was positive or negative
