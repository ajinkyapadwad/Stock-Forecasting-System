# -*- coding: utf-8 -*-
from __future__ import unicode_literals

"""
    APRIL 2017
    (c) Ajinkya
"""
from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect

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




# ------------------------------------------------------------

# ----------------------- REALTIME / HISTORICAL  DATA LOGGING----------------------------
import numpy
import math

db = MySQLdb.connect("127.0.0.1","root","password","realtime" )    # DB 1 for real-time
db2 = MySQLdb.connect("127.0.0.1","root","password","historical" )  # DB for historical data

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
#db.close()
#db2.close()

# --------------------------------------------------------------------------

# ------------------------ BAYESIAN ---------------------------------------------

def passing(request):
    options=request.GET.get('optionsRadios')
    dur = request.GET.get('predict')
    print options
    print dur
    pred = query(options)
    return render(request,'realtime/prediction2_new.html',{'pred':pred}) 
def bayesian(data):
    x_10 =[]
    t_data = []
    for i in range(len(data) - 90, len(data)):
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
    
    return final


def query(name):

    data=[]
    if name is 'amazon':
        sql = "SELECT price FROM Amazon"
    elif name is 'apple':
        sql = "SELECT price FROM Apple"
    elif name is 'facebook':
        sql = "SELECT price FROM Facebook"
    elif name is 'google':
        sql = "SELECT price FROM Google"
    elif name is 'easports':
        sql = "SELECT price FROM EAsports"
    elif name is 'microsoft':
        sql = "SELECT price FROM Microsoft"
    elif name is 'walmart':
        sql = "SELECT price FROM Walmart"
    elif name is 'yahoo':
        sql = "SELECT price FROM Yahoo"
    elif name is 'sony':
        sql = "SELECT price FROM Sony"
    elif name is 'nikon':
        sql = "SELECT price FROM Nikon"

    # Prepare SQL query to INSERT a record into the database.
    cursor2.execute(sql)
    results = cursor2.fetchall()
    for row in results:
        data.append(row[0])
    #print ('price', results)

    prediction = bayesian(data)
    print ("FINAL PREDICTION BAYESIAN : ", prediction)
    return prediction


    


##def newprediction(request):
##    return render(request, 'realtime/newprediction.html', {'content':[pred, 'here']})
# ----------------------------------------------------------------------------

# -----------------------  NEURAL NETWORK ---------------------------------------

# from datetime import datetime
# from time import mktime

# from neuralNetwork import NeuralNetwork

# ## ================================================================

# def normalizePrice(price, minimum, maximum):
#     return ((2*price - (maximum + minimum)) / (maximum - minimum))

# def denormalizePrice(price, minimum, maximum):
#     return (((price*(maximum-minimum))/2) + (maximum + minimum))/2

# ## ================================================================

# def rollingWindow(seq, windowSize):
#     it = iter(seq)
#     win = [it.next() for cnt in xrange(windowSize)] # First window
#     yield win
#     for e in it: # Subsequent windows
#         win[:-1] = win[1:]
#         win[-1] = e
#         yield win

# def getMovingAverage(values, windowSize):
#     movingAverages = []
    
#     for w in rollingWindow(values, windowSize):
#         movingAverages.append(sum(w)/len(w))

#     return movingAverages

# def getMinimums(values, windowSize):
#     minimums = []

#     for w in rollingWindow(values, windowSize):
#         minimums.append(min(w))
            
#     return minimums

# def getMaximums(values, windowSize):
#     maximums = []

#     for w in rollingWindow(values, windowSize):
#         maximums.append(max(w))

#     return maximums


# def getTimeSeriesValues(values, window):
#     movingAverages = getMovingAverage(values, window)
#     minimums = getMinimums(values, window)
#     maximums = getMaximums(values, window)

#     returnData = []

#     # build items of the form [[average, minimum, maximum], normalized price]
#     for i in range(0, len(movingAverages)):
#         inputNode = [movingAverages[i], minimums[i], maximums[i]]
#         price = normalizePrice(values[len(movingAverages) - (i + 1)], minimums[i], maximums[i])
#         outputNode = [price]
#         tempItem = [inputNode, outputNode]
#         returnData.append(tempItem)

#     return returnData


# def getHistoricalData(stockSymbol):
#     historicalPrices = []
    
#     # login to API
#     urllib2.urlopen("http://api.kibot.com/?action=login&user=guest&password=guest")

#     # get 14 days of data from API (business days only, could be < 10)
#     url = "http://api.kibot.com/?action=history&symbol=" + stockSymbol + "&interval=daily&period=14&unadjusted=1&regularsession=1"
#     apiData = urllib2.urlopen(url).read().split("\n")
#     for line in apiData:
#         if(len(line) > 0):
#             tempLine = line.split(',')
#             price = float(tempLine[1])
#             historicalPrices.append(price)

#     return historicalPrices


# def getTrainingData(stockSymbol):
#     historicalData = getHistoricalData(stockSymbol)

#     # reverse it so we're using the most recent data first, ensure we only have 9 data points
#     historicalData.reverse()
#     del historicalData[9:]

#     # get five 5-day moving averages, 5-day lows, and 5-day highs, associated with the closing price
#     trainingData = getTimeSeriesValues(historicalData, 5)

#     return trainingData

# def getPredictionData(stockSymbol):
#     historicalData = getHistoricalData(stockSymbol)

#     # reverse it so we're using the most recent data first, then ensure we only have 5 data points
#     historicalData.reverse()
#     del historicalData[5:]

#     # get five 5-day moving averages, 5-day lows, and 5-day highs
#     predictionData = getTimeSeriesValues(historicalData, 5)
#     # remove associated closing price
#     predictionData = predictionData[0][0]

#     return predictionData


# def analyzeSymbol(stockSymbol):
#     startTime = time.time()
    
#     trainingData = getTrainingData(stockSymbol)
    
#     network = NeuralNetwork(inputNodes = 3, hiddenNodes = 3, outputNodes = 1)

#     network.train(trainingData)

#     # get rolling data for most recent day
#     predictionData = getPredictionData(stockSymbol)

#     # get prediction
#     returnPrice = network.test(predictionData)

#     # de-normalize and return predicted stock price
#     predictedStockPrice = denormalizePrice(returnPrice, predictionData[1], predictionData[2])

#     # create return object, including the amount of time used to predict
#     returnData = {}
#     returnData['price'] = predictedStockPrice
#     returnData['time'] = time.time() - startTime

#     return returnData

# if __name__ == "__main__":
#     print analyzeSymbol("GOOG")

# # ------------------------------------------------------------------------------------------------------

# # ---------------------------------- STATE VECTOR MACHINE -----------------------------------------------

# import pandas
# import numpy as np
# from sklearn import preprocessing
# from sklearn import svm
# from sklearn import cross_validation

# # read the data
# df = pandas.read_csv('techsectordatareal.csv')
# daysAhead = 270

# # calculate price volatility array given company
# def calcPriceVolatility(numDays, priceArray):
# 	global daysAhead
# 	# make price volatility array
# 	volatilityArray = []
# 	movingVolatilityArray = []
# 	for i in range(1, numDays+1):
# 		percentChange = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]
# 		movingVolatilityArray.append(percentChange)
# 	volatilityArray.append(np.mean(movingVolatilityArray))
# 	for i in range(numDays + 1, len(priceArray) - daysAhead):
# 		del movingVolatilityArray[0]
# 		percentChange = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]
# 		movingVolatilityArray.append(percentChange)
# 		volatilityArray.append(np.mean(movingVolatilityArray))

# 	return volatilityArray

# # calculate momentum array
# def calcMomentum(numDays, priceArray):
# 	global daysAhead
# 	# now calculate momentum
# 	momentumArray = []
# 	movingMomentumArray = []
# 	for i in range(1, numDays + 1):
# 		movingMomentumArray.append(1 if priceArray[i] > priceArray[i-1] else -1)
# 	momentumArray.append(np.mean(movingMomentumArray))
# 	for i in range(numDays+1, len(priceArray) - daysAhead):
# 		del movingMomentumArray[0]
# 		movingMomentumArray.append(1 if priceArray[i] > priceArray[i-1] else -1)
# 		momentumArray.append(np.mean(movingMomentumArray))

# 	return momentumArray

# def makeModelAndPredict(permno, numDays, sectorVolatility, sectorMomentum, splitNumber):
# 	global df
# 	global daysAhead
# 	# get price volatility and momentum for this company
# 	companyData = df[df['PERMNO'] == permno]
# 	companyPrices = list(companyData['PRC'])

# 	volatilityArray = calcPriceVolatility(numDays, companyPrices)
# 	momentumArray = calcMomentum(numDays, companyPrices)

# 	splitIndex = splitNumber - numDays

# 	# since they are different lengths, find the min length
# 	if len(volatilityArray) > len(sectorVolatility):
# 		difference = len(volatilityArray) - len(sectorVolatility)
# 		del volatilityArray[:difference]
# 		del momentumArray[:difference]

# 	elif len(sectorVolatility) > len(volatilityArray):
# 		difference = len(sectorVolatility) - len(volatilityArray)
# 		del sectorVolatility[:difference]
# 		del sectorMomentum[:difference]

# 	# create the feature vectors X
# 	X = np.transpose(np.array([volatilityArray, momentumArray, sectorVolatility, sectorMomentum]))

# 	# create the feature vectors Y
# 	Y = []
# 	for i in range(numDays, len(companyPrices) - daysAhead):
# 		Y.append(1 if companyPrices[i+daysAhead] > companyPrices[i] else -1)
# 	print len(Y)

# 	# fix the length of Y if necessary
# 	if len(Y) > len(X):
# 		print 'here2'
# 		difference = len(Y) - len(X)
# 		del Y[:difference]

# 	# split into training and testing sets
# 	X_train = np.array(X[0:splitIndex]).astype('float64')
# 	X_test = np.array(X[splitIndex:]).astype('float64')
# 	y_train = np.array(Y[0:splitIndex]).astype('float64')
# 	y_test = np.array(Y[splitIndex:]).astype('float64')

# 	# fit the model and calculate its accuracy
# 	rbf_svm = svm.SVC(kernel='rbf')
# 	rbf_svm.fit(X_train, y_train)
# 	score = rbf_svm.score(X_test, y_test)
# 	print score
# 	return score

# def main():
# 	global df

# 	# find the list of companies
# 	permnoList = sorted(set(list(df['PERMNO'])))
# 	companiesNotFull = [12084, 13407, 14542, 93002, 15579] # companies without full dates

# 	# read the tech sector data
# 	ndxtdf = pandas.read_csv('ndxtdata.csv')
# 	ndxtdf = ndxtdf.sort_index(by='Date', ascending=True)
# 	ndxtPrices = list(ndxtdf['Close'])

# 	# find when 2012 starts
# 	startOfTwelve = list(df[df['PERMNO'] == 10107]['date']).index(20120103)

# 	# we want to predict where it will be on the next day based on X days previous
# 	numDaysArray = [5, 10, 20, 90, 270] # day, week, month, quarter, year

# 	predictionDict = {}

# 	# iterate over combinations of n_1 and n_2 and find prediction accuracies
# 	for numDayIndex in numDaysArray:
# 		for numDayStock in numDaysArray:
# 			ndxtVolatilityArray = calcPriceVolatility(numDayIndex, ndxtPrices)
# 			ndxtMomentumArray = calcMomentum(numDayIndex, ndxtPrices)
# 			predictionForGivenNumDaysDict = {}

# 			for permno in permnoList:
# 				if permno in companiesNotFull:
# 					continue
# 				print permno
# 				percentage = makeModelAndPredict(permno,numDayStock,ndxtVolatilityArray,ndxtMomentumArray,startOfTwelve)
# 				predictionForGivenNumDaysDict[permno] = percentage


# 			predictionAccuracies = predictionForGivenNumDaysDict.values()
# 			meanAccuracy = np.mean(predictionAccuracies)
# 			maxIndex = max(predictionForGivenNumDaysDict, key=predictionForGivenNumDaysDict.get)
# 			maxAccuracy = (maxIndex, predictionForGivenNumDaysDict[maxIndex])
# 			minIndex = min(predictionForGivenNumDaysDict, key=predictionForGivenNumDaysDict.get)
# 			minAccuracy = (minIndex, predictionForGivenNumDaysDict[minIndex])
# 			median = np.median(predictionAccuracies)

# 			numDaysTuple = (numDayIndex, numDayStock)
# 			predictionDict[numDaysTuple] = {'mean':meanAccuracy, 'max':predictionForGivenNumDaysDict[maxIndex], 'min':predictionForGivenNumDaysDict[minIndex], 'median':median }

# 	sortedTuples = sorted(predictionDict.keys())
# 	for numDaysTuple in sortedTuples:
# 		# print "%s:\t %s\n" % (numDaysTuple, predictionDict[numDaysTuple])
# 		sumStats = predictionDict[numDaysTuple]
# 		print "& %d & %d & %f & %f & %f & %f \\\\\n" % (numDaysTuple[0], numDaysTuple[1], sumStats['mean'], sumStats['median'], sumStats['max'], sumStats['min'])

# if __name__ == "__main__": 
# 	#main()

# # ------------------------------------------------------------------------------------------------------


