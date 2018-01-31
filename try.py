import re
import urllib2
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import numpy as np
import time


a = [1,1,2,3,4,5,3,2,1]

list = dict([(x, a.count(x)) for x in a])
print(list)
print(list[1])
print(list[2])
print(list[3])