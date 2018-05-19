from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import codecs
from os import listdir
from os.path import isfile, join

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')
en_stop.append('yes')
en_stop.append('ye')
en_stop.append('e')
en_stop.append('g')
en_stop.append('will')
en_stop.append('http')
en_stop.append('www')
en_stop.append('et')
en_stop.append('al')
en_stop.append('de')
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

lineperfile = 700
wordperline = 20
# compile sample documents into a list
path = input("Enter the path: ") or "rawfile/"
print(path)
lineperfile = int(input("Enter line per file:") or "600")
print(lineperfile)
wordperline = int(input('Enter word per line:') or '20')
print(wordperline)
doc_set = [f for f in listdir(path) if f.endswith('.txt')]
# loop through document list
for i in doc_set:
       print(i)
       content = []
       with open(path + i,'r',encoding='utf-8') as f:
              for line in f.readlines():
                     #print(line)                     
                     lowraw = line.strip().lower()
                     tokens = tokenizer.tokenize(lowraw)
                     # remove stop words from tokens
                     #stopped_tokens = [i for i in tokens if not i in en_stop]
                     stopped_tokens = tokens
                     # remove ditigs from tokens
                     without_digits = [i for i in stopped_tokens if not i.isdigit()]
                     # stem tokens
                     stemmed_tokens = without_digits
                     #stemmed_tokens = [p_stemmer.stem(i) for i in without_digits]
                     # remove_characterOrOneDigit_tokens
                     stemmed_without_character = [i for i in stemmed_tokens if not (len(i)==1)]
                     for item in stemmed_without_character:
                            content.append(item)
              last = 0
              lines = []
              while last < len(content):
                     line = content[last:last + wordperline]
                     lines.append(line)
                     last += wordperline
              last2 = 0
              part = 0
              while last2 < len(lines):
                     fout = open(i[:i.rindex(".")] + "part" + str(part)+".txt",'w',encoding='utf-8')
                     sublines = lines[last2:last2+lineperfile]
                     for line in sublines:
                            fout.write(" ".join(line))
                            fout.write('\n')
                     part +=1
                     last2 += lineperfile
