# -*- coding: utf-8 -*-
from __future__ import unicode_literals

"""
	APRIL 2017
	(c) Ajinkya
"""
from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect

from django.template import RequestContext

import MySQLdb                      #import SQL DB library
import datetime                     #import date time library
from yahoo_finance import Share     #import yahoo finance library
import time                         #import time library for delay


# --------------- MAIN WEB PAGES -----------------------------
def index(request):
	return render(request, 'realtime/index.html') 

def pred(request):
	return render(request, 'realtime/prediction.html') 

def news(request):
	return render(request, 'realtime/news.html') 

def contact(request):
	return render(request, 'realtime/contact.html') 


# # ----------------------- REALTIME / HISTORICAL  DATA LOGGING----------------------------
import numpy
import math

db = MySQLdb.connect("127.0.0.1","root","","realtime" )    # DB 1 for real-time
db2 = MySQLdb.connect("127.0.0.1","root","","historical" )  # DB for historical data

# cursors for the two DB's
cursor = db.cursor()    
cursor2 = db2.cursor()

# Ticker symbols for stocks - 
A = Share('AMZN')
AP = Share('AAPL')
F = Share('FB')
G = Share('GOOG')
E = Share('EA')
M = Share('MSFT')
W = Share('WMT')
Y = Share('YHOO')
S = Share('SNE')
N = Share('NINOY')

# function to track real time data
def RealTimeStocks():
	print ("Creating tables for Real-time logging ... ")
	# create 10 separate tables - 
	sql = """CREATE TABLE Amazon(date CHAR(40),price FLOAT,volume INT )"""   # SQL query
	cursor.execute(sql)                                                     # execute SQL query
	sql = """CREATE TABLE Apple(date CHAR(40),price FLOAT,volume INT )"""
	cursor.execute(sql)
	sql = """CREATE TABLE Facebook(date CHAR(40),price FLOAT,volume INT )"""
	cursor.execute(sql)
	sql = """CREATE TABLE Google(date CHAR(40),price FLOAT,volume INT )"""
	cursor.execute(sql)
	sql = """CREATE TABLE EASports(date CHAR(40),price FLOAT,volume INT )"""
	cursor.execute(sql)
	sql = """CREATE TABLE Microsoft(date CHAR(40),price FLOAT,volume INT )"""
	cursor.execute(sql)
	sql = """CREATE TABLE Walmart(date CHAR(40),price FLOAT,volume INT )"""
	cursor.execute(sql)
	sql = """CREATE TABLE Yahoo(date CHAR(40),price FLOAT,volume INT )"""
	cursor.execute(sql)
	sql = """CREATE TABLE Sony(date CHAR(40),price FLOAT,volume INT )"""
	cursor.execute(sql)
	sql = """CREATE TABLE Nikon(date CHAR(40),price FLOAT,volume INT )"""
	cursor.execute(sql)


	# loop to track real time data
	for t in range(250):  # = must be set to 505 for tracking 7 hours (10 AM to 5 PM) @ 50 sec refresh rate
		for name in [A,AP,F,G,E,M,W,Y,S,N] :          # scan the five tickers
			date=datetime.datetime.now()   # get current system time
			price= name.get_price()         # get current stock price
			volume= name.get_volume()       # get current volume

			if name is Y:   # insert all above avlues into Yahoo table
				sq2 = "INSERT INTO Yahoo (date, price, volume )VALUES ('%s', '%s', '%s' )" % (date, price, volume)
			if name is G:   # insert into Google table
				sq2 = "INSERT INTO Google (date, price, volume )VALUES ('%s', '%s', '%s' )" % (date, price, volume)
			elif name is A:
				sq2 = "INSERT INTO Amazon (date, price, volume )VALUES ('%s', '%s', '%s' )" % (date, price, volume)
			elif name is E:
				sq2 = "INSERT INTO EASports (date, price, volume )VALUES ('%s', '%s', '%s' )" % (date, price, volume)
			elif name is AP:
				sq2 = "INSERT INTO Apple (date, price, volume )VALUES ('%s', '%s', '%s' )" % (date, price, volume)
			elif name is F:   # insert all above avlues into Yahoo table
				sq2 = "INSERT INTO Facebook(date, price, volume )VALUES ('%s', '%s', '%s' )" % (date, price, volume)
			elif name is M:   # insert into Google table
				sq2 = "INSERT INTO Microsoft(date, price, volume )VALUES ('%s', '%s', '%s' )" % (date, price, volume)
			elif name is W:
				sq2 = "INSERT INTO Walmart(date, price, volume )VALUES ('%s', '%s', '%s' )" % (date, price, volume)
			elif name is S:
				sq2 = "INSERT INTO Sony(date, price, volume )VALUES ('%s', '%s', '%s' )" % (date, price, volume)
			elif name is N:
				sq2 = "INSERT INTO  Nikon(date, price, volume )VALUES ('%s', '%s', '%s' )" % (date, price, volume)

			# else : >> ?


			cursor.execute(sq2) # execute SQL query
			db.commit()    # commit all the changes

		time.sleep(30)   # = delay of 50 seconds expected. 

		for name in [A,AP,F,G,E,M,W,Y,S,N] :
			name.refresh()  # refresh command

# function to log historical data
def HistoricalStocks():
	print ("Creating tables for historical logging ... ")
	# create five separate tables -
	for name in [A,AP,F,G,E,M,W,Y,S,N] :
		if name is Y:
			sql = """CREATE TABLE Yahoo ( date  CHAR(40),open FLOAT,high FLOAT,low FLOAT,price FLOAT,volume INT )"""
		if name is G:
			sql = """CREATE TABLE Google ( date  CHAR(40),open FLOAT,high FLOAT,low FLOAT,price FLOAT,volume INT )"""
		if name is A:
			sql = """CREATE TABLE Amazon ( date  CHAR(40),open FLOAT,high FLOAT,low FLOAT,price FLOAT,volume INT )"""
		if name is E:
			sql = """CREATE TABLE EASports ( date  CHAR(40),open FLOAT,high FLOAT,low FLOAT,price FLOAT,volume INT )"""
		if name is AP:
			sql = """CREATE TABLE Apple ( date  CHAR(40),open FLOAT,high FLOAT,low FLOAT,price FLOAT,volume INT )"""
		if name is F:
			sql = """CREATE TABLE Facebook ( date  CHAR(40),open FLOAT,high FLOAT,low FLOAT,price FLOAT,volume INT )"""
		if name is M:
			sql = """CREATE TABLE Microsoft ( date  CHAR(40),open FLOAT,high FLOAT,low FLOAT,price FLOAT,volume INT )"""
		if name is W:
			sql = """CREATE TABLE Walmart ( date  CHAR(40),open FLOAT,high FLOAT,low FLOAT,price FLOAT,volume INT )"""
		if name is S:
			sql = """CREATE TABLE Sony ( date  CHAR(40),open FLOAT,high FLOAT,low FLOAT,price FLOAT,volume INT )"""
		if name is N:
			sql = """CREATE TABLE Nikon ( date  CHAR(40),open FLOAT,high FLOAT,low FLOAT,price FLOAT,volume INT )"""
	
		cursor2.execute(sql)    # execute SQL query
			   
		#dic = name.get_historical('2015-12-31', '2016-12-31')   # get historical data dictionary
		dic = name.get_historical('2016-01-31', '2016-12-31')   # get historical data dictionary
		
		for i in range(len(dic)):   # scan dictionary data

			val=list(dic[i].values()) # ignore keys, take just values 

			if name is Y:   # insert all above avlues into Yahoo table
				sq2 = "INSERT INTO Yahoo  (date, open, high, low, price, volume )VALUES ('%s', '%s', '%s', '%s', '%s', '%s' )" % (val[5], val[2], val[4], val[3], val[7], val[0])
			if name is G:   # insert all above avlues into Google table
				sq2 = "INSERT INTO Google (date, open, high, low, price, volume )VALUES ('%s', '%s', '%s', '%s', '%s', '%s' )" % (val[5], val[2], val[4], val[3], val[7], val[0])
			if name is A:      
				sq2 = "INSERT INTO Amazon  (date, open, high, low, price, volume )VALUES ('%s', '%s', '%s', '%s', '%s', '%s' )" % (val[5], val[2], val[4], val[3], val[7], val[0])
			if name is E:      
				sq2 = "INSERT INTO EASports  (date, open, high, low, price, volume )VALUES ('%s', '%s', '%s', '%s', '%s', '%s' )" % (val[5], val[2], val[4], val[3], val[7], val[0])
			if name is AP:      
				sq2 = "INSERT INTO Apple (date, open, high, low, price, volume )VALUES ('%s', '%s', '%s', '%s', '%s', '%s' )" % (val[5], val[2], val[4], val[3], val[7], val[0])
			if name is F:   # insert all above avlues into Yahoo table
				sq2 = "INSERT INTO Facebook  (date, open, high, low, price, volume )VALUES ('%s', '%s', '%s', '%s', '%s', '%s' )" % (val[5], val[2], val[4], val[3], val[7], val[0])
			if name is M:   # insert all above avlues into Google table
				sq2 = "INSERT INTO Microsoft (date, open, high, low, price, volume )VALUES ('%s', '%s', '%s', '%s', '%s', '%s' )" % (val[5], val[2], val[4], val[3], val[7], val[0])
			if name is W:      
				sq2 = "INSERT INTO Walmart  (date, open, high, low, price, volume )VALUES ('%s', '%s', '%s', '%s', '%s', '%s' )" % (val[5], val[2], val[4], val[3], val[7], val[0])
			if name is S:      
				sq2 = "INSERT INTO Sony  (date, open, high, low, price, volume )VALUES ('%s', '%s', '%s', '%s', '%s', '%s' )" % (val[5], val[2], val[4], val[3], val[7], val[0])
			if name is N:      
				sq2 = "INSERT INTO Nikon (date, open, high, low, price, volume )VALUES ('%s', '%s', '%s', '%s', '%s', '%s' )" % (val[5], val[2], val[4], val[3], val[7], val[0])
			
			cursor2.execute(sq2)    # execute SQL query

			db2.commit()    # commit all the changes

# Call the two functions
#RealTimeStocks()
#HistoricalStocks()

# close databases
# db.close()
# db2.close()

# --------------------------------------------------------------------------

# ------------------------ BAYESIAN ---------------------------------------------



def Bayesian(data,days):
	x_10 =[]
	t_data = []
	for i in range(len(data) - (100 - int(days)), len(data)):
		t_data.append(data[i])
	for i in range(1, 11):
		x_10.append(i)
	t=[]
	t.append(t_data)
	t_data = t
	N = 10
	M = 6

	x=x_10[len(x_10) - 1] + 1

	for k in range(1):
		t = numpy.zeros((N,1),float)
		phi = numpy.zeros((M,1),float)
		phi_sum = numpy.zeros((M,1),float)
		phi_sum_t = numpy.zeros((M,1),float)

		for i in range(M):
			phi[i][0]=math.pow(x,i)

		for i in range(N):
		   t[i][0]=t_data[k][i]
		
		#calculation of phi_ sum and phi_sum_t       
		for j in range(N):
			for i in range(M):
				phi_sum[i][0]=phi_sum[i][0]+math.pow(x_10[j],i)
				phi_sum_t[i][0]=phi_sum_t[i][0]+t[j][0]*math.pow(x_10[j],i)

	# Calculation of variance / standard deviation
		S=numpy.linalg.inv(0.005*numpy.identity(M)+11.1*numpy.dot(phi_sum,phi.T))

		var=numpy.dot((phi.T),numpy.dot(S,phi))
		var=var+1/11.1
	# Calculating the mean
		mean=11.1*numpy.dot(phi.T,numpy.dot(S,phi_sum_t))
	   
		mean = mean[0][0]
		#print ' Predicted Price(bayesian mean)', mean

	t = t_data[0]
	t_data = t
	sum = 0
	avg = 0
	for i in t_data:
		sum += i
	avg = sum / len(t_data)
	#print 'mov', avg
	per = ((mean - avg) / avg) * 100
	#print 'error % = ', per
	final = []
	mean = round(mean, 3)
	per = round(per, 3)
	final.append(mean)
	final.append(per)
	
	return mean

dataset=[]
def CallBayesian(name,days):
	# print(name)
	data=[]
	if name in ('amzn'):
		sql = "SELECT price FROM Amazon"
	elif name in ('aapl'):
		sql = "SELECT price FROM Apple"
	elif name in ('fb'):
		sql = "SELECT price FROM Facebook"
	elif name in ('goog'):
		sql = "SELECT price FROM Google"
	elif name in ('ea'):
		sql = "SELECT price FROM EAsports"
	elif name in ('msft'):
		sql = "SELECT price FROM Microsoft"
	elif name in ('wmt'):
		sql = "SELECT price FROM Walmart"
	elif name in ('yhoo'):
		sql = "SELECT price FROM Yahoo"
	elif name in ('sne'):
		sql = "SELECT price FROM Sony"
	elif name in ('ninoy'):
		sql = "SELECT price FROM Nikon"

	cursor2.execute(sql)
	results = cursor2.fetchall()
	for row in results:
		data.append(row[0])
	#print ('price', results)
	dataset = data
	prediction = Bayesian(data,days)

	# print ("FINAL PREDICTION BAYESIAN : ", prediction)
	return prediction

# ----------------------------------------------------------------------------

# -----------------------  NEURAL NETWORK ---------------------------------------

from datetime import datetime
import math, random, string
from time import mktime
import urllib2


random.seed(0)

# calculate a random number a <= rand < b
def rand(a, b):
	return (b-a)*random.random() + a

def makeMatrix(I, J, fill = 0.0):
	m = []
	for i in range(I):
		m.append([fill]*J)
	return m

def sigmoid(x):
	# tanh is a little nicer than the standard 1/(1+e^-x)
	return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
	return 1.0 - y**2


class NeuralNetwork:
	def __init__(self, inputNodes, hiddenNodes, outputNodes):
		# number of input, hidden, and output nodes
		self.inputNodes = inputNodes + 1 # +1 for bias node
		self.hiddenNodes = hiddenNodes
		self.outputNodes = outputNodes

		# activations for nodes
		self.inputActivation = [1.0]*self.inputNodes
		self.hiddenActivation = [1.0]*self.hiddenNodes
		self.outputActivation = [1.0]*self.outputNodes
		
		# create weights
		self.inputWeight = makeMatrix(self.inputNodes, self.hiddenNodes)
		self.outputWeight = makeMatrix(self.hiddenNodes, self.outputNodes)
		# set them to random vaules
		for i in range(self.inputNodes):
			for j in range(self.hiddenNodes):
				self.inputWeight[i][j] = rand(-0.2, 0.2)
		for j in range(self.hiddenNodes):
			for k in range(self.outputNodes):
				self.outputWeight[j][k] = rand(-2.0, 2.0)

		# last change in weights for momentum   
		self.ci = makeMatrix(self.inputNodes, self.hiddenNodes)
		self.co = makeMatrix(self.hiddenNodes, self.outputNodes)

	def update(self, inputs):
		if len(inputs) != self.inputNodes-1:
			raise ValueError('wrong number of inputs')

		# input activations
		for i in range(self.inputNodes-1):
			self.inputActivation[i] = inputs[i]

		# hidden activations
		for j in range(self.hiddenNodes):
			sum = 0.0
			for i in range(self.inputNodes):
				sum = sum + self.inputActivation[i] * self.inputWeight[i][j]
			self.hiddenActivation[j] = sigmoid(sum)

		# output activations
		for k in range(self.outputNodes):
			sum = 0.0
			for j in range(self.hiddenNodes):
				sum = sum + self.hiddenActivation[j] * self.outputWeight[j][k]
			self.outputActivation[k] = sigmoid(sum)

		return self.outputActivation[:]


	def backPropagate(self, targets, N, M):
		if len(targets) != self.outputNodes:
			raise ValueError('wrong number of target values')

		# calculate error terms for output
		output_deltas = [0.0] * self.outputNodes
		for k in range(self.outputNodes):
			error = targets[k]-self.outputActivation[k]
			output_deltas[k] = dsigmoid(self.outputActivation[k]) * error

		# calculate error terms for hidden
		hidden_deltas = [0.0] * self.hiddenNodes
		for j in range(self.hiddenNodes):
			error = 0.0
			for k in range(self.outputNodes):
				error = error + output_deltas[k]*self.outputWeight[j][k]
			hidden_deltas[j] = dsigmoid(self.hiddenActivation[j]) * error

		# update output weights
		for j in range(self.hiddenNodes):
			for k in range(self.outputNodes):
				change = output_deltas[k]*self.hiddenActivation[j]
				self.outputWeight[j][k] = self.outputWeight[j][k] + N*change + M*self.co[j][k]
				self.co[j][k] = change

		# update input weights
		for i in range(self.inputNodes):
			for j in range(self.hiddenNodes):
				change = hidden_deltas[j]*self.inputActivation[i]
				self.inputWeight[i][j] = self.inputWeight[i][j] + N*change + M*self.ci[i][j]
				self.ci[i][j] = change

		# calculate error
		error = 0.0
		for k in range(len(targets)):
			error = error + 0.5*(targets[k] - self.outputActivation[k])**2
			
		return error


	def test(self, inputNodes):
		print(inputNodes, '->', self.update(inputNodes))
		return self.update(inputNodes)[0]

	def weights(self):
		#print('Input weights:')
		for i in range(self.inputNodes):
			print(self.inputWeight[i])
		#print()
		#print('Output weights:')
		for j in range(self.hiddenNodes):
			print(self.outputWeight[j])

	def train(self, patterns, iterations = 1000, N = 0.5, M = 0.1):
		# N: learning rate, M: momentum factor
		for i in range(iterations):
			error = 0.0
			for p in patterns:
				inputs = p[0]
				targets = p[1]
				self.update(inputs)
				error = error + self.backPropagate(targets, N, M)
			# if i % 100 == 0:
			# 	print('error %-.5f' % error)

   
def normalizePrice(price, minimum, maximum):
	return ((2*price - (maximum + minimum)) / (maximum - minimum))

def denormalizePrice(price, minimum, maximum):
	return (((price*(maximum-minimum))/2) + (maximum + minimum))/2



def rollingWindow(seq, windowSize):
	it = iter(seq)
	win = [it.next() for cnt in xrange(windowSize)] # First window
	yield win
	for e in it: # Subsequent windows
		win[:-1] = win[1:]
		win[-1] = e
		yield win

MO = []
def getMovingAverage(values, windowSize):
	movingAverages = []
	
	for w in rollingWindow(values, windowSize):
		movingAverages.append(sum(w)/len(w))

	MO = movingAverages
	return movingAverages

def getMinimums(values, windowSize):
	minimums = []

	for w in rollingWindow(values, windowSize):
		minimums.append(min(w))
			
	return minimums

def getMaximums(values, windowSize):
	maximums = []

	for w in rollingWindow(values, windowSize):
		maximums.append(max(w))

	return maximums


def getTimeSeriesValues(values, window):
	movingAverages = getMovingAverage(values, window)
	minimums = getMinimums(values, window)
	maximums = getMaximums(values, window)

	returnData = []

	# build items of the form [[average, minimum, maximum], normalized price]
	for i in range(0, len(movingAverages)):
		inputNode = [movingAverages[i], minimums[i], maximums[i]]
		price = normalizePrice(values[len(movingAverages) - (i + 1)], minimums[i], maximums[i])
		outputNode = [price]
		tempItem = [inputNode, outputNode]
		returnData.append(tempItem)

	return returnData


def getHistoricalData(stockSymbol):
	historicalPrices = []
	
	# login to API
	urllib2.urlopen("http://api.kibot.com/?action=login&user=guest&password=guest")

	# get 14 days of data from API (business days only, could be < 10)
	url = "http://api.kibot.com/?action=history&symbol=" + stockSymbol + "&interval=daily&period=14&unadjusted=1&regularsession=1"
	apiData = urllib2.urlopen(url).read().split("\n")
	for line in apiData:
		if(len(line) > 0):
			tempLine = line.split(',')
			price = float(tempLine[1])
			historicalPrices.append(price)

	return historicalPrices


def getTrainingData(stockSymbol):
	historicalData = getHistoricalData(stockSymbol)

	# reverse it so we're using the most recent data first, ensure we only have 9 data points
	historicalData.reverse()
	del historicalData[9:]

	# get five 5-day moving averages, 5-day lows, and 5-day highs, associated with the closing price
	trainingData = getTimeSeriesValues(historicalData,5)

	return trainingData

def getPredictionData(stockSymbol):
	historicalData = getHistoricalData(stockSymbol)

	# reverse it so we're using the most recent data first, then ensure we only have 5 data points
	historicalData.reverse()
	del historicalData[5:]

	# get five 5-day moving averages, 5-day lows, and 5-day highs
	predictionData = getTimeSeriesValues(historicalData, 5)
	# remove associated closing price
	predictionData = predictionData[0][0]

	return predictionData


def CallNeural(stockSymbol):
	if stockSymbol in ('amazon'):
		stockSymbol = "AMZN"
	if stockSymbol in ('yahoo'):
		stockSymbol = "YHOO"
	if stockSymbol in ('apple'):
		stockSymbol = "AAPL"
	if stockSymbol in ('facebook'):
		stockSymbol = "FB"
	if stockSymbol in ('google'):
		stockSymbol = "GOOG"
	if stockSymbol in ('easports'):
		stockSymbol = "EA"
	if stockSymbol in ('microsoft'):
		stockSymbol = "MSFT"
	if stockSymbol in ('walmart'):
		stockSymbol = "WMT"
	if stockSymbol in ('sony'):
		stockSymbol = "SNE"
	if stockSymbol in ('nikon'):
		stockSymbol = "NINOY"
	
	startTime = time.time()
	
	trainingData = getTrainingData(stockSymbol)
	
	network = NeuralNetwork(inputNodes = 3, hiddenNodes = 3, outputNodes = 1)

	network.train(trainingData)

	# get rolling data for most recent day
	predictionData = getPredictionData(stockSymbol)

	# get prediction
	returnPrice = network.test(predictionData)

	# de-normalize and return predicted stock price
	predictedStockPrice = denormalizePrice(returnPrice, predictionData[1], predictionData[2])

	# create return object, including the amount of time used to predict
	returnData = {}
	returnData['price'] = predictedStockPrice
	returnData['time'] = time.time() - startTime

	return predictionData[1]



# # ------------------------------------------------------------------------------------------------------
 
# # ---------------------------------- SUPPORT VECTOR MACHINE -----------------------------------------------

import scipy
from scipy.stats import norm
from sklearn.svm import SVR, SVC, LinearSVC

def CallSVM(name, day):
	'''
	Input: Name of the stock, How many days of after current day to get predicted price
	Output: Predicted Price for next n days
	'''
	# data = getHistoricalData(name)
	data=[]
	if name in ('amzn'):
		sql = "SELECT price FROM Amazon"
	elif name in ('aapl'):
		sql = "SELECT price FROM Apple"
	elif name in ('fb'):
		sql = "SELECT price FROM Facebook"
	elif name in ('goog'):
		sql = "SELECT price FROM Google"
	elif name in ('ea'):
		sql = "SELECT price FROM EAsports"
	elif name in ('msft'):
		sql = "SELECT price FROM Microsoft"
	elif name in ('wmt'):
		sql = "SELECT price FROM Walmart"
	elif name in ('yhoo'):
		sql = "SELECT price FROM Yahoo"
	elif name in ('sne'):
		sql = "SELECT price FROM Sony"
	elif name in ('ninoy'):
		sql = "SELECT price FROM Nikon"

	cursor2.execute(sql)
	results = cursor2.fetchall()
	for row in results:
		data.append(row[0])

	data = data[::-1]
	open_price_list = []
	close_price_list = []
	predicted_price=[]
	for i in range(len(data)):
		open_price_list.append(data[i][1])
		close_price_list.append(data[i][2])
	for iterations in range(day):
		close_price_dataset=[]
		open_price_dataset=[]
		previous_ten_day_close_price_dataset=[]
		g=0
		h=50
		while h<len(close_price_list):
			previous_ten_day_close_price_dataset.append(close_price_list[g:h])
			open_price_dataset.append(open_price_list[h])
			close_price_dataset.append(close_price_list[h])
			g += 1
			h += 1
		moving_average_dataset=[]
		for x in previous_ten_day_close_price_dataset:
			i=0
			for y in x:
				i=i+y
			moving_average_dataset.append(i/10)
		feature_dataset = []
		for j in range(len(close_price_dataset)):
			list = []
			list.append(moving_average_dataset[j])
			list.append(open_price_dataset[j])
			feature_dataset.append(list)
		feature_dataset = numpy.array(feature_dataset)        
		close_price_dataset = numpy.array(close_price_dataset)
		clf = SVR(kernel='linear',degree=1)
		clf.fit(feature_dataset[-365:],close_price_dataset[-365:])
		target = []
		if iterations==0:
			url_string = "http://www.google.com/finance/getprices?q={0}".format(name)
			stock_info = Share(name)
			list = []
			list.append(stock_info.get_open())
			list.append(stock_info.get_50day_moving_avg())
			target.append(list)
			
		else:
			list = []
			list.append(moving_average_dataset[-1])
			list.append(open_price_dataset[-1])
			target.append(list)

		predicted_close_price = clf.predict(target)[0]
		predicted_price.append(predicted_close_price)
		open_price_list.append(close_price_list[-1])
		close_price_list.append(predicted_close_price)
	
	return predicted_price

# # ------------------------------------   INDICATORS ------------------------------------------------------------------


def RSI(prices):
	# RSI is calculated using a period of 14 days
	period = 14
	# Range is one period less than the amount of prices input
	data_range = len(prices) - period
	# If there are less than 14 prices, the RSI cannot be calculated, and the system exits
	if data_range < 0:
		raise SystemExit

	# Calculates the daily price change
	price_change = prices[1:] - prices[:-1]
	# An array of zeros the length of data_range is created
	rsi = np.zeros(data_range)

	# Creates an array with the price changes
	gains = np.array(price_change)
	# Only the positive values will be kept in the gains array
	negative_gains = gains < 0
	gains[negative_gains] = 0

	# Creates an array of losses where only the negative values are kept, and then multiplied by -1 for the next step
	losses = np.array(price_change)
	positive_gains = gains > 0
	losses[positive_gains] = 0
	losses *=-1

	# Calculate the mean of the up days and the down days
	avg_up = np.mean(gains[:period])
	avg_down = np.mean(losses[:period])

	if avg_down == 0:
		rsi[0] = 100
	else:
		RS = avg_up/avg_down
		rsi[0] = 100 - (100/(1+RS))

	for i in range(1,data_range):
		avg_up = (avg_up * (period-1) + gains[i + (period - 1)])/ \
				period
		avg_down = (avg_down * (period-1) + losses[i + (period - 1)])/ \
				period

		if avg_down == 0:
			rsi[i] = 100
		else:
			RS = avg_up/avg_down
			rsi[i] = 100 - (100/(1+RS))

	return rsi

def MO():
	print MO
	return MO[1]

## ---------------------------------- QUERIES -----------------------------------------------------------

# Latest price for given stock
def GetLatestPrice(name):
	print(name)
	data=[]
	if name in ('amzn'):
		sql = "SELECT price FROM Amazon LIMIT 1"
	elif name in ('aapl'):
		sql = "SELECT price FROM Apple LIMIT 1"
	elif name in ('fb'):
		sql = "SELECT price FROM Facebook LIMIT 1"
	elif name in ('goog'):
		sql = "SELECT price FROM Google LIMIT 1"
	elif name in ('ea'):
		sql = "SELECT price FROM EAsports LIMIT 1"
	elif name in ('msft'):
		sql = "SELECT price FROM Microsoft LIMIT 1"
	elif name in ('wmt'):
		sql = "SELECT price FROM Walmart LIMIT 1"
	elif name in ('yhoo'):
		sql = "SELECT price FROM Yahoo LIMIT 1"
	elif name in ('sne'):
		sql = "SELECT price FROM Sony LIMIT 1"
	elif name in ('ninoy'):
		sql = "SELECT price FROM Nikon LIMIT 1"

	cursor2.execute(sql)
	results = cursor2.fetchall()
	for row in results:
		data.append(row[0])
	#print ('price', results)

	#prediction = bayesian(data)
	#prediction = 
	print ("Returns the latest price of each company : ", results)
	return round(results[0][0],2)

# highest price in past 10 days
def GetMaximumPrice(name):
	print(name)
	data=[]
	if name in ('amzn'):
		sql = "SELECT MAX(price) FROM Amazon LIMIT 10"
	elif name in ('aapl'):
		sql = "SELECT MAX(price) FROM Apple LIMIT 10"
	elif name in ('fb'):
		sql = "SELECT MAX(price) FROM Facebook LIMIT 10"
	elif name in ('goog'):
		sql = "SELECT MAX(price) FROM Google LIMIT 10"
	elif name in ('ea'):
		sql = "SELECT MAX(price) FROM EAsports LIMIT 10"
	elif name in ('msft'):
		sql = "SELECT MAX(price) FROM Microsoft LIMIT 10"
	elif name in ('wmt'):
		sql = "SELECT MAX(price) FROM Walmart LIMIT 10"
	elif name in ('yhoo'):
		sql = "SELECT MAX(price) FROM Yahoo LIMIT 10"
	elif name in ('sne'):
		sql = "SELECT MAX(price) FROM Sony LIMIT 10"
	elif name in ('ninoy'):
		sql = "SELECT MAX(price) FROM Nikon LIMIT 10"

	cursor2.execute(sql)
	results = cursor2.fetchall()
	for row in results:
		data.append(row[0])
	#print ('price', results)

	#prediction = bayesian(data)
	#prediction = 
	print ("Returns the max value of last 10 days : ", results)
	return round(results[0][0],2)

# lowest price in past one year
def GetMinimumPrice(name):
	print(name)
	data=[]
	if name in ('amzn'):
		sql = "SELECT MIN(price) FROM Amazon LIMIT 365"
	elif name in ('aapl'):
		sql = "SELECT MIN(price) FROM Apple LIMIT 365"
	elif name in ('fb'):
		sql = "SELECT MIN(price) FROM Facebook LIMIT 365"
	elif name in ('goog'):
		sql = "SELECT MIN(price) FROM Google LIMIT 365"
	elif name in ('easports'):
		sql = "SELECT MIN(price) FROM EAsports LIMIT 365"
	elif name in ('msft'):
		sql = "SELECT MIN(price) FROM Microsoft LIMIT 365"
	elif name in ('wmt'):
		sql = "SELECT MIN(price) FROM Walmart LIMIT 365"
	elif name in ('yhoo'):
		sql = "SELECT MIN(price) FROM Yahoo LIMIT 365"
	elif name in ('sne'):
		sql = "SELECT MIN(price) FROM Sony LIMIT 365"
	elif name in ('ninoy'):
		sql = "SELECT MIN(price) FROM Nikon LIMIT 365"

	cursor2.execute(sql)
	results = cursor2.fetchall()
	for row in results:
		data.append(row[0])
	#print ('price', results)

	#prediction = bayesian(data)
	#prediction = 
	print ("Returns the max value of last 10 days : ", results)
	return round(results[0][0],0)

# average price in past one year
def GetAveragePrice(name):
	print(name)
	data=[]
	if name in ('amzn'):
		sql = "SELECT AVG(price) FROM Amazon LIMIT 365"
	elif name in ('aapl'):
		sql = "SELECT AVG(price) FROM Apple LIMIT 365"
	elif name in ('fb'):
		sql = "SELECT AVG(price) FROM Facebook LIMIT 365"
	elif name in ('google'):
		sql = "SELECT AVG(price) FROM Google LIMIT 365"
	elif name in ('easports'):
		sql = "SELECT AVG(price) FROM EAsports LIMIT 365"
	elif name in ('msft'):
		sql = "SELECT AVG(price) FROM Microsoft LIMIT 365"
	elif name in ('wmt'):
		sql = "SELECT AVG(price) FROM Walmart LIMIT 365"
	elif name in ('yhoo'):
		sql = "SELECT AVG(price) FROM Yahoo LIMIT 365"
	elif name in ('sne'):
		sql = "SELECT AVG(price) FROM Sony LIMIT 365"
	elif name in ('ninoy'):
		sql = "SELECT AVG(price) FROM Nikon LIMIT 365"

	cursor2.execute(sql)
	results = cursor2.fetchall()
	for row in results:
		data.append(row[0])

	print ("Returns the avg value of last 1yr : ", results)
	return round(results[0][0],2)

# list of companies with average less than selected stock
def GetBelowAverage(name):

	data=[]
	if name in ('amzn'):
		# sql = "SELECT MIN(price) FROM Amazon LIMIT 365"
		# cursor2.execute(sql)
		# results = cursor2.fetchall()
		GetMinimumPrice('amzn')
		for company in ('amzn','aapl','fb','google','easports','msft','wmt','yhoo','sne','ninoy'):
			if (GetAveragePrice(company) < GetMinimumPrice('amzn')):
				data.append(company)

	elif name in ('aapl'):
		# sql = "SELECT MIN(price) FROM Apple LIMIT 365"
		# cursor2.execute(sql)
		# results = cursor2.fetchall()
		for company in ('amzn','aapl','fb','google','easports','msft','wmt','yhoo','sne','ninoy'):
			if (GetAveragePrice(company) < GetMinimumPrice('aapl')):
				data.append(company)

	elif name in ('fb'):
		# sql = "SELECT MIN(price) FROM Facebook LIMIT 365"
		# cursor2.execute(sql)
		# results = cursor2.fetchall()
		for company in ('amzn','aapl','fb','google','easports','msft','wmt','yhoo','sne','ninoy'):
			if (GetAveragePrice(company) < GetMinimumPrice('fb')):
				data.append(company)

	elif name in ('google'):
		# sql = "SELECT MIN(price) FROM Google LIMIT 365"
		# cursor2.execute(sql)
		# results = cursor2.fetchall()
		for company in ('amzn','aapl','fb','google','easports','msft','wmt','yhoo','sne','ninoy'):
			if (GetAveragePrice(company) < GetMinimumPrice('google')):
				data.append(company)

	elif name in ('easports'):
		# sql = "SELECT MIN(price) FROM EAsports LIMIT 365"
		# cursor2.execute(sql)
		# results = cursor2.fetchall()
		for company in ('amzn','aapl','fb','google','easports','msft','wmt','yhoo','sne','ninoy'):
			if (GetAveragePrice(company) < GetMinimumPrice('easports')):
				data.append(company)

	elif name in ('msft'):
		# sql = "SELECT MIN(price) FROM Microsoft LIMIT 365"
		# cursor2.execute(sql)
		# results = cursor2.fetchall()
		for company in ('amzn','aapl','fb','google','easports','msft','wmt','yhoo','sne','ninoy'):
			if (GetAveragePrice(company) < GetMinimumPrice('msft')):
				data.append(company)

	elif name in ('wmt'):
		# sql = "SELECT MIN(price) FROM Walmart LIMIT 365"
		# cursor2.execute(sql)
		# results = cursor2.fetchall()
		for company in ('amzn','aapl','fb','google','easports','msft','wmt','yhoo','sne','ninoy'):
			if (GetAveragePrice(company) < GetMinimumPrice('wmt')):
				data.append(company)

	elif name in ('yhoo'):
		# sql = "SELECT MIN(price) FROM Yahoo LIMIT 365"
		# cursor2.execute(sql)
		# results = cursor2.fetchall()
		for company in ('amzn','aapl','fb','google','easports','msft','wmt','yhoo','sne','ninoy'):
			if (GetAveragePrice(company) < GetMinimumPrice('yhoo')):
				data.append(company)

	elif name in ('sne'):
		# sql = "SELECT MIN(price) FROM Sony LIMIT 365"
		# cursor2.execute(sql)
		# results = cursor2.fetchall()
		for company in ('amzn','aapl','fb','google','easports','msft','wmt','yhoo','sne','ninoy'):
			if (GetAveragePrice(company) < GetMinimumPrice('sne')):
				data.append(company)

	elif name in ('ninoy'):
		# sql = "SELECT MIN(price) FROM Nikon LIMIT 365"
		# cursor2.execute(sql)
		# results = cursor2.fetchall()
		for company in ('amzn','aapl','fb','google','easports','msft','wmt','yhoo','sne','ninoy'):
			if (GetAveragePrice(company) < GetMinimumPrice('ninoy')):
				data.append(company)

	# print data
	# return data

	finaldata = []
	for name in data:
		if name in ('amzn'):
			finaldata.append('Amazon')
		if name in ('aapl'):
			finaldata.append('Apple')
		if name in ('fb'):
			finaldata.append('Facebook')
		if name in ('goog'):
			finaldata.append('Google')
		if name in ('ea'):
			finaldata.append('EASports')
		if name in ('msft'):
			finaldata.append('Microsoft')
		if name in ('wmt'):
			finaldata.append('Walmart')
		if name in ('yhoo'):
			finaldata.append('Yahoo')
		if name in ('sne'):
			finaldata.append('Sony')
		if name in ('ninoy'):
			finaldata.append('Nikon')

	print finaldata
	return finaldata

# # ---------------------------------------------------------------------
def search(request,name):

	
	baye1 = CallBayesian(name,10)
	baye2 = CallBayesian(name,10)
	baye3 = CallBayesian(name,10)

	baye4 = CallBayesian(name,10)
	baye5 = CallBayesian(name,10)
	baye6 = CallBayesian(name,10)
	
	baye7 = CallBayesian(name,10)
	baye8 = CallBayesian(name,10)
	baye9 = CallBayesian(name,10)

	neu1 = CallNeural(name)
	neu1 = neu1 - (0.1 * neu1)
	neu2 = CallNeural(name)
	neu2 = neu2 - (0.1 * neu2)
	neu3 = CallNeural(name)
	neu3 = neu3 - (0.1 * neu3)
	neu4 = CallNeural(name)
	neu4 = neu4 - (0.1 * neu4)
	neu5 = CallNeural(name)
	neu5 = neu5 - (0.1 * neu5)
	neu6 = CallNeural(name)
	neu6 = neu6 - (0.1 * neu6)
	neu7 = CallNeural(name)
	neu7 = neu7 - (0.1 * neu7)
	neu8 = CallNeural(name)
	neu8 = neu8 - (0.1 * neu8)
	neu9 = CallNeural(name)
	neu9 = neu9 - (0.1 * neu9)

	# svm1 = CallSVM(name,10)
	# svm2 = CallSVM(name,10)
	# svm3 = CallSVM(name,10)
	# svm4 = CallSVM(name,10)
	# svm5 = CallSVM(name,10)
	# svm6 = CallSVM(name,10)
	# svm7 = CallSVM(name,10)
	# svm8 = CallSVM(name,10)
	# svm9 = CallSVM(name,10)

	# pred1= CallBayesian(name,10)
	# pred2 = CallNeural(name)
	# pred2 = pred2 - (0.1 * pred2)
	#pred3 = CallSVM(name,10)
	pred3 = 0

	#RSIndex = RSI(dataset)
	RSIndex = 0

	#MA = MO()
	MA = 4444
	# THIRD INDICATOR

	LatestPrice = GetLatestPrice(name)
	MaximumPrice = GetMaximumPrice(name)
	MinimumPrice = GetMinimumPrice(name)
	AveragePrice = GetAveragePrice(name)
	BelowAverage = GetBelowAverage(name)
	#BelowAverage = 0
	return render(request, 'realtime/search.html',
		{
			'name':name, 
			# 'pred1':pred1, 
			# 'pred2':pred2, 

			'baye1':baye1,
			'baye2':baye2,
			'baye3':baye3,
			'baye4':baye4,
			'baye5':baye5,
			'baye6':baye6,
			'baye7':baye7,
			'baye8':baye8,
			'baye9':baye9,

			'neu1':neu1,
			'neu2':neu2,
			'neu3':neu3,
			'neu4':neu4,
			'neu5':neu5,
			'neu6':neu6,
			'neu7':neu7,
			'neu8':neu8,
			'neu9':neu9,


			'pred3':pred3, 
			'RSIndex':RSIndex, 
			'MA':MA,
			'LatestPrice' :LatestPrice,
			'MaximumPrice':MaximumPrice,
			'MinimumPrice':MinimumPrice,
			'AveragePrice':AveragePrice,
			'BelowAverage':BelowAverage
		})