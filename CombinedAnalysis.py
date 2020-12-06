import sqlite3
from sqlite3 import Error

import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.sparse import hstack

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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

	pd.options.mode.chained_assignment = None  # default='warn'

	db_path = "/Users/lijun/Desktop/CS591B2/project/filteredData/myYelpResData.db"
	conn = create_connection(db_path)
	conn.text_factory = lambda b: b.decode(errors='ignore')

	# get restaurant filtered reviews & regular reviews
	combined_list = run_query(conn, "SELECT * FROM combined WHERE flagged IN ('Y', 'N')")
	combined_col_names = refine_col_names(run_query(conn, "PRAGMA table_info('combined')"))

	# turn list to dataframe
	combined = pd.DataFrame(combined_list, columns=combined_col_names)

	print("combined shape : ", combined.shape)
	print("combined head :\n", combined.head())
	print("\n\n\n")

	# get data we may want to use
	cols = ['reviewContent', 'rating', 'flagged',
			'friendCount', 'reviewCount', 'firstCount', 'fanCount', 'reviewer_flagged',
			'posReviewCount', 'negReviewCount'
			]
	data = combined[cols]

	print("data shape : ", data.shape)
	print("data head :\n", data.head())
	print("\n\n\n")

	# cleanup data
	data['flagged'] = (data['flagged'] == 'Y').astype(int)
	data['reviewer_flagged'] = (data['reviewer_flagged'] == 'Y').astype(int)
	data['ratio'] = data['posReviewCount'] / data['negReviewCount']
	data['ratio'] = data['ratio'].replace(np.inf, 0)
	data['ratio'] = data['ratio'].replace(np.nan, 0)

	print("data shape : ", data.shape)
	print("data head :\n", data.head())
	print("\n\n\n")

	# check for infinite/null data
	check = []
	for i in cols:
		if (i != 'reviewContent'):
			check.append(np.isfinite(data[i]).all())
	print("check if all data are finite : ", check)
	print("check if null data exist", data.isnull().sum())
	print("\n\n\n")

	# calculate then plot and print correlation matrix
	# create a correlation matrix that measures the linear relationships between the variables
	correlation_matrix = data.corr().round(2)
	# annot = True to print the values inside the square
	sns.heatmap(correlation_matrix, annot=True)
	plt.show()

	print("correlation matrix :\n", correlation_matrix)
	print("\n\n\n")

	# choosing data we are going to use
	usable_data = data.drop(['rating', 'fanCount', 'posReviewCount', 'negReviewCount'], axis=1)
	print("usable_data shape : ", usable_data.shape)
	print("usable_data head :\n", usable_data.head())
	print("\n\n\n")

	# preprocess data
	stop_words = stopwords.words('english')
	lemmatizer = WordNetLemmatizer()
	wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

	usable_data['reviewContent'] = usable_data['reviewContent'].apply(clean_text)

	print("preprocessing text data :")
	print("usable_data shape : ", usable_data.shape)
	print("usable_data head :\n", usable_data.head(), "\n\n\n")

	numerical_cols = ['friendCount', 'reviewCount', 'firstCount', 'ratio']

	for i in numerical_cols:
		scale = StandardScaler().fit(data[[i]])
		data[i] = scale.transform(data[[i]])

	print("preprocessing numerical data :")
	print("usable_data shape is : ", usable_data.shape)
	print("usable_data head :\n", usable_data.head(), "\n\n\n")

	# set X and y
	X = usable_data.drop('flagged', axis=1)
	y = usable_data['flagged']

	# split into training set and testing set
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

	# count vectorizor
	count_vectorizor = CountVectorizer()
	count_vectorizor.fit(X_train['reviewContent'])
	X_cv = count_vectorizor.transform(X_train['reviewContent'])
	X_cv_test = count_vectorizor.transform(X_test['reviewContent'])

	cv_stack = hstack([X_cv, X_train[['friendCount', 'reviewCount', 'firstCount', 'ratio']].values],
								format="csr")
	cv_stack_test = hstack([X_cv_test, X_test[['friendCount', 'reviewCount', 'firstCount', 'ratio']].values],
									 format="csr")

	# count vectorizor ngram
	count_ngram_vectorizor = CountVectorizer(ngram_range=(1, 3))
	count_ngram_vectorizor.fit(X_train['reviewContent'])
	X_cv_ngram = count_ngram_vectorizor.transform(X_train['reviewContent'])
	X_cv_ngram_test = count_ngram_vectorizor.transform(X_test['reviewContent'])

	cv_ngram_stack = hstack(
		[X_cv_ngram, X_train[['friendCount', 'reviewCount', 'firstCount', 'ratio']].values], format="csr")
	cv_ngram_stack_test = hstack(
		[X_cv_ngram_test, X_test[['friendCount', 'reviewCount', 'firstCount', 'ratio']].values], format="csr")

	# tf-idf vectorizor
	tfidf_vectorizor = TfidfVectorizer()
	tfidf_vectorizor.fit(X_train['reviewContent'])
	X_tv = tfidf_vectorizor.transform(X_train['reviewContent'])
	X_tv_test = tfidf_vectorizor.transform(X_test['reviewContent'])

	tv_stack = hstack([X_tv, X_train[['friendCount', 'reviewCount', 'firstCount', 'ratio']].values],
								format="csr")
	tv_stack_test = hstack([X_tv_test, X_test[['friendCount', 'reviewCount', 'firstCount', 'ratio']].values],
									 format="csr")

	# tf-idf vectorizor ngram
	tfidf_ngram_vectorizor = TfidfVectorizer(ngram_range=(1, 3))
	tfidf_ngram_vectorizor.fit(X_train['reviewContent'])
	X_tv_ngram = tfidf_ngram_vectorizor.transform(X_train['reviewContent'])
	X_tv_ngram_test = tfidf_ngram_vectorizor.transform(X_test['reviewContent'])

	tv_ngram_stack = hstack([X_tv, X_train[['friendCount', 'reviewCount', 'firstCount', 'ratio']].values],
									  format="csr")
	tv_ngram_stack_test = hstack(
		[X_tv_test, X_test[['friendCount', 'reviewCount', 'firstCount', 'ratio']].values], format="csr")

	# logistic regression w/ count vectorizor
	print("\n\n\nlogistic regression w/ count vectorizor :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		lr_cv = LogisticRegression(C=c, max_iter=30000)
		lr_cv.fit(cv_stack, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, lr_cv.predict(cv_stack_test))))

	# logistic regression w/ count vectorizor ngram
	print("\n\n\nlogistic regression w/ count vectorizor ngram :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		lr_cv_ngram = LogisticRegression(C=c, max_iter=30000)
		lr_cv_ngram.fit(cv_ngram_stack, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, lr_cv_ngram.predict(cv_ngram_stack_test))))

	# logistic regression w/ tf-idf vectorizor
	print("\n\n\nlogistic regression w/ tf-idf vectorizor :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		lr_tv = LogisticRegression(C=c, max_iter=30000)
		lr_tv.fit(tv_stack, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, lr_tv.predict(tv_stack_test))))

	# logistic regression w/ tf-idf vectorizor ngram
	print("\n\n\nlogistic regression w/ tf-idf vectorizor ngram :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		lr_tv_ngram = LogisticRegression(C=c, max_iter=30000)
		lr_tv_ngram.fit(tv_ngram_stack, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, lr_tv_ngram.predict(tv_ngram_stack_test))))

	# linear SVM w/ count vectorizor
	print("\n\n\nlinear SVM w/ count vectorizor :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		svm_cv = LinearSVC(C=c, max_iter=30000)
		svm_cv.fit(cv_stack, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, svm_cv.predict(cv_stack_test))))

	# linear SVM w/ count vectorizor ngram
	print("\n\n\nlinear SVM w/ count vectorizor ngram :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		svm_cv_ngram = LinearSVC(C=c, max_iter=30000)
		svm_cv_ngram.fit(cv_ngram_stack, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, svm_cv_ngram.predict(cv_ngram_stack_test))))

	# linear SVM w/ tf-idf vectorizor
	print("\n\n\nlinear SVM w/ tf-idf vectorizor :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		svm_tv = LinearSVC(C=c, max_iter=30000)
		svm_tv.fit(tv_stack, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, svm_tv.predict(tv_stack_test))))

	# linear SVM w/ tf-idf vectorizor ngram
	print("\n\n\nlinear SVM w/ tf-idf vectorizor ngram :\n")
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
		svm_tv_ngram = LinearSVC(C=c, max_iter=30000)
		svm_tv_ngram.fit(tv_ngram_stack, y_train)
		print("Accuracy for C = %s : %s" % (c, accuracy_score(y_test, svm_tv_ngram.predict(tv_ngram_stack_test))))

"""
combined shape :  (26958, 25)
combined head :
         date                reviewID  ... posReviewCount negReviewCount
0  9/22/2012  GtwU21YOQn-wf4vWRUIx6w  ...             29              9
1  9/22/2012                 0LpVTc3  ...             11              0
2  9/19/2012           tljtLzf68Fkwf  ...             11              0
3   9/6/2012                     iSN  ...             53              5
4   9/9/2012                  Jmwrh7  ...            629            231

[5 rows x 25 columns]




data shape :  (26958, 10)
data head :
                                        reviewContent  ...  negReviewCount
0  Unlike Next, which we'd eaten at the previous ...  ...               9
1  Probably one of the best meals I've had ever. ...  ...               0
2  Service was impeccable. Experience and present...  ...               0
3  The problem with places like this, given the e...  ...               5
4  I have no idea how to write my review - dining...  ...             231

[5 rows x 10 columns]




data shape :  (26958, 11)
data head :
                                        reviewContent  ...      ratio
0  Unlike Next, which we'd eaten at the previous ...  ...   3.222222
1  Probably one of the best meals I've had ever. ...  ...   0.000000
2  Service was impeccable. Experience and present...  ...   0.000000
3  The problem with places like this, given the e...  ...  10.600000
4  I have no idea how to write my review - dining...  ...   2.722944

[5 rows x 11 columns]




check if all data are finite :  [True, True, True, True, True, True, True, True, True]
check if null data exist reviewContent       0
rating              0
flagged             0
friendCount         0
reviewCount         0
firstCount          0
fanCount            0
reviewer_flagged    0
posReviewCount      0
negReviewCount      0
ratio               0
dtype: int64




correlation matrix :
                   rating  flagged  ...  negReviewCount  ratio
rating              1.00    -0.06  ...           -0.03   0.11
flagged            -0.06     1.00  ...           -0.23  -0.24
friendCount         0.01    -0.11  ...            0.49   0.12
reviewCount         0.01    -0.23  ...            0.84   0.18
firstCount          0.02    -0.11  ...            0.49   0.15
fanCount            0.01    -0.10  ...            0.49   0.10
reviewer_flagged   -0.06     0.99  ...           -0.23  -0.24
posReviewCount      0.03    -0.22  ...            0.76   0.23
negReviewCount     -0.03    -0.23  ...            1.00   0.01
ratio               0.11    -0.24  ...            0.01   1.00

[10 rows x 10 columns]




usable_data shape :  (26958, 7)
usable_data head :
                                        reviewContent  ...      ratio
0  Unlike Next, which we'd eaten at the previous ...  ...   3.222222
1  Probably one of the best meals I've had ever. ...  ...   0.000000
2  Service was impeccable. Experience and present...  ...   0.000000
3  The problem with places like this, given the e...  ...  10.600000
4  I have no idea how to write my review - dining...  ...   2.722944

[5 rows x 7 columns]




preprocessing text data :
usable_data head :
                                        reviewContent  ...      ratio
0  unlike next wed eaten previous night dish comp...  ...   3.222222
1  probably one best meal ive ever performance fo...  ...   0.000000
2  service impeccable experience presentation coo...  ...   0.000000
3  problem place like this give exhorbitant cost ...  ...  10.600000
4  idea write review din alinea brings whole diff...  ...   2.722944

[5 rows x 7 columns] 



usable_data shape is :  (26958, 7)
preprocessing numerical data :
usable_data head :
                                        reviewContent  ...      ratio
0  unlike next wed eaten previous night dish comp...  ...   3.222222
1  probably one best meal ive ever performance fo...  ...   0.000000
2  service impeccable experience presentation coo...  ...   0.000000
3  problem place like this give exhorbitant cost ...  ...  10.600000
4  idea write review din alinea brings whole diff...  ...   2.722944

[5 rows x 7 columns] 



usable_data shape is :  (26958, 7)



logistic regression w/ count vectorizor :

Accuracy for C = 0.01 : 0.8529673590504451
Accuracy for C = 0.05 : 0.8491097922848665
Accuracy for C = 0.25 : 0.8396142433234421
Accuracy for C = 0.5 : 0.8314540059347181
Accuracy for C = 1 : 0.8231454005934719



logistic regression w/ count vectorizor ngram :

Accuracy for C = 0.01 : 0.8529673590504451
Accuracy for C = 0.05 : 0.8458456973293769
Accuracy for C = 0.25 : 0.8339762611275965
Accuracy for C = 0.5 : 0.8304154302670623
Accuracy for C = 1 : 0.8283382789317507



logistic regression w/ tf-idf vectorizor :

Accuracy for C = 0.01 : 0.8480712166172106
Accuracy for C = 0.05 : 0.8554896142433235
Accuracy for C = 0.25 : 0.8606824925816023
Accuracy for C = 0.5 : 0.8583086053412463
Accuracy for C = 1 : 0.8541543026706232



logistic regression w/ tf-idf vectorizor ngram :

Accuracy for C = 0.01 : 0.8480712166172106
Accuracy for C = 0.05 : 0.8554896142433235
Accuracy for C = 0.25 : 0.8606824925816023
Accuracy for C = 0.5 : 0.8583086053412463
Accuracy for C = 1 : 0.8541543026706232



linear SVM w/ count vectorizor :

Accuracy for C = 0.01 : 0.8350148367952522
Accuracy for C = 0.05 : 0.8149851632047478
Accuracy for C = 0.25 : 0.7908011869436202
Accuracy for C = 0.5 : 0.7801186943620178
Accuracy for C = 1 : 0.7724035608308605



linear SVM w/ count vectorizor ngram :

Accuracy for C = 0.01 : 0.8307121661721069
Accuracy for C = 0.05 : 0.8176557863501484
Accuracy for C = 0.25 : 0.8080118694362017
Accuracy for C = 0.5 : 0.8060830860534125
Accuracy for C = 1 : 0.8008902077151335



linear SVM w/ tf-idf vectorizor :

Accuracy for C = 0.01 : 0.85
Accuracy for C = 0.05 : 0.8465875370919881
Accuracy for C = 0.25 : 0.8385756676557864
Accuracy for C = 0.5 : 0.8338278931750742
Accuracy for C = 1 : 0.8299703264094955



linear SVM w/ tf-idf vectorizor ngram :

Accuracy for C = 0.01 : 0.849406528189911
Accuracy for C = 0.05 : 0.8465875370919881
Accuracy for C = 0.25 : 0.8370919881305638
Accuracy for C = 0.5 : 0.8320474777448071
Accuracy for C = 1 : 0.8289317507418398
"""