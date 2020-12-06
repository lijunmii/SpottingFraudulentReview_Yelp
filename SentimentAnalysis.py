import sqlite3
from sqlite3 import Error

import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

pd.options.mode.chained_assignment = None  # default='warn'

def create_connection(db_file):
	"""
	create a database connection to a SQLite database specified by the db_file
	:param db_file: path to database file
	:return: Connection object or None
	"""
	conn = None
	try:
		conn = sqlite3.connect(db_file)
	except Error as e:
		print(e)
	return conn


def run_query(conn, query):
	"""
	query all rows in the tasks table
	:param conn: the Connection object
	:param query: query used on db
	:return:
	"""
	cur = conn.cursor()
	cur.execute(query)
	rows = cur.fetchall()
	return rows


def get_X_and_Y(conn, query):
	"""
	query all rows in the tasks table, and get X and Y
	:param conn: the Connection object
	:param query: query used on db
	:return X: features
	:return Y: flagged
	"""
	cur = conn.cursor()
	cur.execute(query)
	rows = cur.fetchall()
	X = []
	Y = []
	for row in rows:
		X.append(row[:8] + row[9:])
		Y.append(row[8])
	return X, Y


def refine_col_names(list):
	names = []
	for col_info in list:
		names.append(col_info[1])
	return names


def clean_text(text):
	# cast characters to lower case
	lower = text.lower()
	# remove stopwords
	no_stop_words = " ".join([word for word in str(lower).split() if word not in stop_words])
	# remove punctuation
	no_punc = no_stop_words.translate(str.maketrans('', '', string.punctuation))
	# lemmatization
	pos_tagged_text = nltk.pos_tag(no_punc.split())
	lammatized = " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
	# lammatized = " ".join([lemmatizer.lemmatize(word) for word in no_punc.split()])
	return lammatized


def close_connection(conn):
	"""
	close database connection
	"""
	if conn:
		conn.close()


if __name__ == '__main__':
	# connect to db
	db_path = "/Users/lijun/Desktop/CS591B2/project/filteredData/myYelpResData.db"
	conn = create_connection(db_path)
	conn.text_factory = lambda b: b.decode(errors='ignore')

	# get restaurant filtered reviews & regular reviews
	reviews_list = run_query(conn, "SELECT * FROM my_review_table WHERE flagged IN ('Y', 'N')")
	col_names = refine_col_names(run_query(conn, "PRAGMA table_info('my_review_table')"))

	# turn list to dataframe
	reviews = pd.DataFrame(reviews_list, columns=col_names)
	print("reviews head :\n", reviews.head(), "\n\n\n")

	# get the data we are going to use
	data = reviews[['reviewContent', 'flagged']]
	data['flagged'] = data['flagged'].replace({'Y': 1, 'N': 0})
	print("data head :\n", data.head(), "\n\n\n")

	# preprocess data
	stop_words = stopwords.words('english')
	lemmatizer = WordNetLemmatizer()
	wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

	data['reviewContent'] = data['reviewContent'].apply(clean_text)
	print("data head :\n", data.head(), "\n\n\n")
	print("data shape is : ", data.shape)

	# split reviews into training set and testing set
	X_train, X_test, y_train, y_test = train_test_split(data['reviewContent'], data['flagged'], train_size=0.75)

	# count vectorizor
	count_vectorizor = CountVectorizer()
	count_vectorizor.fit(X_train)
	X_cv = count_vectorizor.transform(X_train)
	X_cv_test = count_vectorizor.transform(X_test)

	# count vectorizor ngram
	count_ngram_vectorizor = CountVectorizer(ngram_range=(1, 3))
	count_ngram_vectorizor.fit(X_train)
	X_cv_ngram = count_ngram_vectorizor.transform(X_train)
	X_cv_ngram_test = count_ngram_vectorizor.transform(X_test)

	# tf-idf vectorizor
	tfidf_vectorizor = TfidfVectorizer()
	tfidf_vectorizor.fit(X_train)
	X_tv = tfidf_vectorizor.transform(X_train)
	X_tv_test = tfidf_vectorizor.transform(X_test)

	# tf-idf vectorizor ngram
	tfidf_ngram_vectorizor = TfidfVectorizer(ngram_range=(1, 3))
	tfidf_ngram_vectorizor.fit(X_train)
	X_tv_ngram = tfidf_ngram_vectorizor.transform(X_train)
	X_tv_ngram_test = tfidf_ngram_vectorizor.transform(X_test)

	# logistic regression w/ count vectorizor
	print("\n\n\nlogistic regression w/ count vectorizor :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		lr_cv = LogisticRegression(C=c, max_iter=50000)
		lr_cv.fit(X_cv, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, lr_cv.predict(X_cv_test))))

	# logistic regression w/ count vectorizor ngram
	print("\n\n\nlogistic regression w/ count vectorizor ngram :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		lr_cv_ngram = LogisticRegression(C=c, max_iter=50000)
		lr_cv_ngram.fit(X_cv_ngram, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, lr_cv_ngram.predict(X_cv_ngram_test))))

	# logistic regression w/ tf-idf vectorizor
	print("\n\n\nlogistic regression w/ tf-idf vectorizor :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		lr_tv = LogisticRegression(C=c, max_iter=50000)
		lr_tv.fit(X_tv, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, lr_tv.predict(X_tv_test))))

	# logistic regression w/ tf-idf vectorizor ngram
	print("\n\n\nlogistic regression w/ tf-idf vectorizor ngram :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		lr_tv_ngram = LogisticRegression(C=c, max_iter=50000)
		lr_tv_ngram.fit(X_tv_ngram, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, lr_tv_ngram.predict(X_tv_ngram_test))))

	# linear SVM w/ count vectorizor
	print("\n\n\nlinear SVM w/ count vectorizor :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		svm_cv = LinearSVC(C=c, max_iter=50000)
		svm_cv.fit(X_cv, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, svm_cv.predict(X_cv_test))))

	# linear SVM w/ count vectorizor ngram
	print("\n\n\nlinear SVM w/ count vectorizor ngram :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		svm_cv_ngram = LinearSVC(C=c, max_iter=50000)
		svm_cv_ngram.fit(X_cv_ngram, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, svm_cv_ngram.predict(X_cv_ngram_test))))

	# linear SVM w/ tf-idf vectorizor
	print("\n\n\nlinear SVM w/ tf-idf vectorizor :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		svm_tv = LinearSVC(C=c, max_iter=50000)
		svm_tv.fit(X_tv, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, svm_tv.predict(X_tv_test))))

	# linear SVM w/ tf-idf vectorizor ngram
	print("\n\n\nlinear SVM w/ tf-idf vectorizor ngram :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		svm_tv_ngram = LinearSVC(C=c, max_iter=50000)
		svm_tv_ngram.fit(X_tv_ngram, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, svm_tv_ngram.predict(X_tv_ngram_test))))

	close_connection(conn)

"""
output:

reviews head :
         date                reviewID  ... flagged            restaurantID
0  9/22/2012  GtwU21YOQn-wf4vWRUIx6w  ...       N  pbEiXam9YJL3neCYHGwLUA
1  9/22/2012                 0LpVTc3  ...       N  pbEiXam9YJL3neCYHGwLUA
2  9/19/2012           tljtLzf68Fkwf  ...       N  pbEiXam9YJL3neCYHGwLUA
3   9/6/2012                     iSN  ...       N  pbEiXam9YJL3neCYHGwLUA
4   9/9/2012                  Jmwrh7  ...       N  pbEiXam9YJL3neCYHGwLUA

[5 rows x 10 columns] 



data head :
                                        reviewContent  flagged
0  Unlike Next, which we'd eaten at the previous ...        0
1  Probably one of the best meals I've had ever. ...        0
2  Service was impeccable. Experience and present...        0
3  The problem with places like this, given the e...        0
4  I have no idea how to write my review - dining...        0 



data head :
                                        reviewContent  flagged
0  unlike next wed eaten previous night dish comp...        0
1  probably one best meal ive ever performance fo...        0
2  service impeccable experience presentation coo...        0
3  problem place like this give exhorbitant cost ...        0
4  idea write review din alinea brings whole diff...        0 



data shape is :  (67019, 2)



logistic regression w/ count vectorizor :

Accuracy for C = 0.01 : 0.8800954938824231
Accuracy for C = 0.05 : 0.8799164428528797
Accuracy for C = 0.25 : 0.8744255446135482
Accuracy for C = 0.5 : 0.8710832587287377
Accuracy for C = 1 : 0.8664279319606087



logistic regression w/ count vectorizor ngram :

Accuracy for C = 0.01 : 0.8805132796180245
Accuracy for C = 0.05 : 0.8805729632945389
Accuracy for C = 0.25 : 0.8798567591763653
Accuracy for C = 0.5 : 0.8786034019695613
Accuracy for C = 1 : 0.8772903610862429



logistic regression w/ tf-idf vectorizor :

Accuracy for C = 0.01 : 0.8797373918233363
Accuracy for C = 0.05 : 0.8797373918233363
Accuracy for C = 0.25 : 0.8796777081468219
Accuracy for C = 0.5 : 0.8798567591763653
Accuracy for C = 1 : 0.8796777081468219



logistic regression w/ tf-idf vectorizor ngram :

Accuracy for C = 0.01 : 0.8797373918233363
Accuracy for C = 0.05 : 0.8797373918233363
Accuracy for C = 0.25 : 0.8797373918233363
Accuracy for C = 0.5 : 0.8796777081468219
Accuracy for C = 1 : 0.8795583407937929



linear SVM w/ count vectorizor :

Accuracy for C = 0.01 : 0.8800358102059087
Accuracy for C = 0.05 : 0.8740077588779469
Accuracy for C = 0.25 : 0.8587287376902417
Accuracy for C = 0.5 : 0.8485228290062667
Accuracy for C = 1 : 0.8357505222321695



linear SVM w/ count vectorizor ngram :

Accuracy for C = 0.01 : 0.8792599224112205
Accuracy for C = 0.05 : 0.8743658609370337
Accuracy for C = 0.25 : 0.8660698299015219
Accuracy for C = 0.5 : 0.8619516562220233
Accuracy for C = 1 : 0.8575350641599523



linear SVM w/ tf-idf vectorizor :

Accuracy for C = 0.01 : 0.8797373918233363
Accuracy for C = 0.05 : 0.8797970754998508
Accuracy for C = 0.25 : 0.8800358102059087
Accuracy for C = 0.5 : 0.8788421366756192
Accuracy for C = 1 : 0.8767532079976127



linear SVM w/ count vectorizor ngram :

Accuracy for C = 0.01 : 0.8797373918233363
Accuracy for C = 0.05 : 0.8797373918233363
Accuracy for C = 0.25 : 0.8800954938824231
Accuracy for C = 0.5 : 0.8802745449119665
Accuracy for C = 1 : 0.8805729632945389

"""