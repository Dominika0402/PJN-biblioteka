import re
import urllib2
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import numpy as np
import time


stemmer = LancasterStemmer()

#time.sleep(t)



#########################################################ZBIERANIE DANYCH####################################################################
#czyszczenie pliku
open('test.txt', 'w').write("")
open('ocena.txt', 'w').write("")
open('opinia.txt', 'w').write("")
#open('test.html', 'w').write("")

#otwieranie plikow
path = 'opinie_linki.txt'
file = open(path,'r')

path_ocena = 'ocena.txt'
file = open(path_ocena,'w')

path_opinia = 'opinia.txt'
file = open(path_opinia,'w')

linki = []
opinia = []
ocena = []
training_data = []

linia = 0

with open(path) as f:
    lines = f.readlines()
    linki.append(lines)

for link1 in linki:
	#print(linia)
	for link2 in link1:
		
		response =urllib2.urlopen(link2)
		page_source = response.read()

		page3 = re.findall(r"<span class=\"review-score-count\">(.*)</span>.+[\n]*[\t]*.*<span class=\"review-time\">.*<time datetime=\"[0-9\- :]*\".*</time></span>.+[\n]*[\t]*.*</div>",page_source)
		page4 = re.findall(r"<p class=\"product-review-body\">(.*\n*[<br/>]*.*\n*[<br/>]*.*)</p>.*[\t]*.*[\n]*.*[\t]*<div.*",page_source)
		

		for word_ocena in page3:
			page = re.sub(r"<br/>", "", word_ocena)
			#open('./ocena.txt', 'a+').write(page)
			#open('./ocena.txt', 'a+').write("\n")
			ocena.append(page)


		#print("ocena: ", len(ocena))

		for word_opinia in page4:
			page = re.sub(r"<br/>|\n", "", word_opinia)
			#open('./opinia.txt', 'a+').write(page)
			#open('./opinia.txt', 'a+').write("\n")
			opinia.append(page)
		#print("opinia: ", len(opinia))
	linia += 1

#open('./opinia2.txt', 'a+').write(opinia)
#open('./ocena2.txt', 'a+').write(ocena)

licznik = 0

while (licznik<len(ocena)-20):
	training_data.append({"class":ocena[licznik], "sentence":opinia[licznik]})
	licznik += 1;


print ("%s sentences in training data" % len(training_data))
#print ("%s opinia: " % len(opinia))


############################################wydobywanie klas, poszczegolnych klas, dokumentow(para: slowo, klasa)#######################################################

words = []
classes = []
documents = []
ignore_words = ['?']
for pattern in training_data:
    source = unicode(pattern['sentence'], 'iso 8859-2')
    w = nltk.word_tokenize(source)
    words.extend(w)
    documents.append((w, pattern['class']))
    
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

#uzycie stem do tworzenia podstawowej formy slow
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# usuwanie duplikatow
classes = list(set(classes))

#wypisywanie dlugosci dokumentu, klas, slow
print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)


###############################################################################################################################################

training = []
output = []
output_empty = [0] * len(classes)

# zbior trenujacy - tworzenie paczki slow dla kazdego zdania
for doc in documents:
    bag = []
    pattern_words = doc[0]	#wybieramy zdanie z listy documents
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]	#zamiana duzych liter na male
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)	#jedynka jesli w z words znajduje sie w pattern_words

    training.append(bag)
 
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1		#output - 1 jesli zdanie nalezy do danej klasy, 0 jesli do niej nie nalezy
    output.append(output_row)

i = 0
w = documents[i][0]
print ([stemmer.stem(word.lower()) for word in w])
print ("training: ",training[i])
print ("output: ",output[i])

##########################################################################################################################################

# funkcja liniowa sigmoidalna - uzyta do znormalizowania wartocci i jej pochodnej do pomiaru wspolczynnika bledu. Iterowanie i dostosowywanie do momentu, gdy nasz poziom bledu jest akceptowalnie niski.
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# zamiana wyjscia funkcji na pochodna
def sigmoid_output_to_derivative(output):
    return output*(1-output)
 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

#zwracanie ciagu 0 i 1 w zaleznosci od tego czy dane slowo istnieje w zdaniu czy nie
def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)

    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)		#jesli showdetails == true wypisuje zdanie i tablice
    # warstwa wejsciowa jest nasza paczka slow
    l0 = x
    # warstwa druga to macierz wymnozenia wejscia i warstwy ukrytej
    l1 = sigmoid(np.dot(l0, synapse_0))
    # warstwa wyjsciowa
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2


################################tworzenie wag###########################################

def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

	#X - training, y - output
    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)

    last_mean_error = 1

    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
        
    for j in iter(range(epochs+1)):
    	#print(j)

        # przesylanie przez warstwy 0, 1, 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))		#dot roznoznaczny macierzy multiplikacji
                
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # strata wartosci docelowej
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
        	#jesli blad 10000 iteracji jest wiekszy niz ostatniej iteracji -> koniec
        	#mean -> oblicza srednia
        	#abs -> zwraca modul liczby
        	#last_mean_error -> ostatni sredni blad
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
                
     
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)
        layer_1_error = layer_2_delta.dot(synapse_1.T)
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
        
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
	print ("saved synapses to:", synapse_file)

#############################################################################################################################################


X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=20, alpha=0.1, epochs=10000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")


# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])

def classify(sentence, show_details=False):
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    return return_results

#classify("swietna ksiazka")
#classify("polecam")
#classify("Beznadziejna")
#classify("Nie warta uwagi")
#classify("Szkoda czasu i pieniedzy")
#classify("Przesylka nie doszla")
#classify("Nie mam ksiazki, nie moge wystawic opinii o ksiazce, ktorej nie dostarczono jeszcze")
#classify("Wszystko w porzadku", show_details=True)

#########################################TEST####################################################################

sentence2 = opinia[-20:]
mark = ocena[-20:]
good = 0
bad = 0


list = []

for sent in sentence2:
	source = unicode(sent, 'iso 8859-2')
	list.append(classify(source))

k = 0

marks = []
for i in list:
	for j in i:
		for t in j:
			if k%2==0:
				marks.append(t)
			k += 1
####print(marks)	#-> oceny przypuszczalne
####print(mark)		#-> oceny wlasciwe



#k = 0
#for i in marks:
#	if marks[k] == mark[k]:
#		good += 1
#	else:
#		bad += 1
#	k += 1
#print("good: ", good, "bad: ", bad)


#precision -> liczba dobrze odgadnietych przez liczbe wszystkich odgadnietych####################################################
#wrzucenie do tablicy tylko dobrze trafionych ocen
k = 0
dobre = []
for i in marks:
	if marks[k] == mark[k]:
		dobre.append(i)
	k += 1
###########

dobre_dict = dict([x,dobre.count(x)] for x in set(dobre))
dobre_dict_keys = dobre_dict.keys()
dobre_dict_values = dobre_dict.values()
#print(dobre_dict)


list = dict([x,mark.count(x)] for x in set(mark))
list2 = dict([(x, marks.count(x)) for x in marks])

count = []

#list -> opinie pobrane z internetu
#list2 -> opinie wyliczone przez program
list_keys = list.keys()
list_values = list.values()
#print(list_keys)

list2_keys = list2.keys()

#keys -> oceny
#values -> suma wystapien danej oceny
#print("list2_keys ", list2_keys)
list2_values = list2.values()
#print("list2_values ", list2_values)

for i in list2_keys:
	if i in list_keys:
		k = list2_keys.index(i)
		k2 = list_keys.index(i)
		print(i, list2_values[k])		#i -> ocena, list2_values[k] -> suma znalezionych(przez wyliczenie) ocen
		print(i, list_values[k2])		#i -> ocena, list_values[k2] -> suma znalezionych(na stronie internetowej) ocen
		if i in dobre_dict_keys:
			k3 = dobre_dict_keys.index(i)		#szukanie indexu oceny i w tablicy dobrze odnalezionych ocen
		#dobre_dict_values[k3]				#np suma dobrych trafien przy szukaniu oceny 5/5
		#list2_values[k]					#np suma znalezionych ocen 5/5 (poprzez wyliczenia programu)
		#list_values[k2]					#np suma znalezionych ocen 5/5 (na stronie internetowej)
		#precision
			print ("precision dla ",i,": ",str(dobre_dict_values[k3])+"/"+str(list2_values[k]))
			#recall###########################################################################
			print ("recall dla ",i,": ",str(dobre_dict_values[k3])+"/"+str(list_values[k2]))
		else:
			print ("precision dla ",i,": ", 0)
			print ("recall dla ",i,": ",0)		