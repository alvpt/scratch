import nltk

try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

try:
    from nltk.stem import WordNetLemmatizer, PorterStemmer
except:
    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer, PorterStemmer

try:
    from nltk.tokenize import RegexpTokenizer
except:
    nltk.download('omw-1.4')
    from nltk.tokenize import RegexpTokenizer

    
import re


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)


def training_data(sentence):

    bigrams = []

    for i in range(len(sentence)-1):
        bigram_i = [sentence[i], sentence[i+1]] 
        bigrams.append(bigram_i)
    
    return bigrams