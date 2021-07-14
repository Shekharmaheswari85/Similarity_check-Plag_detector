# !pip install PyPDF2
import PyPDF2
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from random import randint
import nltk.data
import spacy
def pdftotext(path):
    pdfFileObj = open(path, 'rb')
    # creating a pdf reader object
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    # printing number of pages in pdf file
    print(pdfReader.numPages)
    # creating a page object
    pageObj = pdfReader.getPage(0)
    # extracting text from page
    DOC1=pageObj.extractText()
    # closing the pdf file object
    pdfFileObj.close()
    return DOC1

def plagcheck(text):

    nlp = spacy.load('en_core_web_sm')
    # Load a text file if required
    text = text
    output = ""

    # Load the pretrained neural net
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # Tokenize the text
    tokenized = tokenizer.tokenize(text)

    # Get the list of words from the entire text
    words = word_tokenize(text)
    nltk.download('averaged_perceptron_tagger')
    # Identify the parts of speech
    tagged = nltk.pos_tag(words)
    doc2=[]
    # doc2.append(words)
    for last in range(0,len(tokenized)):
        for i in range(0,len(words)):
            replacements = []

            # Only replace nouns with nouns, vowels with vowels etc.
            for syn in wordnet.synsets(words[i]):

                # Do not attempt to replace proper nouns or determiners
                if tagged[i][1] == 'NNP' or tagged[i][1] == 'DT':
                    break
                
                # The tokenizer returns strings like NNP, VBP etc
                # but the wordnet synonyms has tags like .n.
                # So we extract the first character from NNP ie n
                # then we check if the dictionary word has a .n. or not 
                word_type = tagged[i][1][0].lower()
                if syn.name().find("."+word_type+"."):
                    # extract the word only
                    r = syn.name()[0:syn.name().find(".")]
                    replacements.append(r)

            if len(replacements) > 0:
                # Choose a random replacement
                replacement = replacements[randint(0,len(replacements)-1)]
                output = output + " " + replacement
            else:
                # If no replacement could be found, then just use the
                # original word
                output = output + " " + words[i]
        # doc2.append(output)
        text1=nlp(text)
        text2=nlp(output)
        ratio=text1.similarity(text2)
        doc2.append(ratio)
    print(max(doc2))
