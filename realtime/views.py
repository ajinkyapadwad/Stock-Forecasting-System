"""
    APRIL 2017
    (c) Ajinkya
"""
from django.shortcuts import render
from django.http import HttpResponse


import MySQLdb                      #import SQL DB library
import datetime                     #import date time library
from yahoo_finance import Share     #import yahoo finance library
import time                         #import time library for delay


# --------------- MAIN WEB PAGES -----------------------------
def index(request):
    #return HttpResponse("Mischief managed.")
    return render(request, 'realtime/index.html') 

def prediction(request):
    return render(request, 'realtime/prediction.html') 

def news(request):
    return render(request, 'realtime/news.html') 

def contact(request):
    return render(request, 'realtime/contact.html') 


# ------------------------------------------------------------

# ----------------------- REALTIME / HISTORICAL  DATA LOGGING----------------------------

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
#db.close()
#db2.close()

# --------------------------------------------------------------------------

# ------------------------ BAYESIAN ---------------------------------------------

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
    if name is A:
        sql = "SELECT price FROM Amazon"
    elif name is AP:
        sql = "SELECT price FROM Apple"
    elif name is F:
        sql = "SELECT price FROM Facebook"
    elif name is G:
        sql = "SELECT price FROM Google"
    elif name is E:
        sql = "SELECT price FROM EAsports"
    elif name is M:
        sql = "SELECT price FROM Microsoft"
    elif name is W:
        sql = "SELECT price FROM Walmart"
    elif name is Y:
        sql = "SELECT price FROM Yahoo"
    elif name is S:
        sql = "SELECT price FROM Sony"
    elif name is N:
        sql = "SELECT price FROM Nikon"

    # Prepare SQL query to INSERT a record into the database.
    cursor2.execute(sql)
    results = cursor2.fetchall()
    for row in results:
        data.append(row[0])
    print ('price', results)

    prediction = bayesian(data)
    print ("FINAL PREDICTION BAYESIAN : ", prediction)

#query(F)
# ----------------------------------------------------------------------------

# -----------------------  NEURAL NETWORK ---------------------------------------

from datetime import datetime
from time import mktime

from neuralNetwork import NeuralNetwork

## ================================================================

def normalizePrice(price, minimum, maximum):
    return ((2*price - (maximum + minimum)) / (maximum - minimum))

def denormalizePrice(price, minimum, maximum):
    return (((price*(maximum-minimum))/2) + (maximum + minimum))/2

## ================================================================

def rollingWindow(seq, windowSize):
    it = iter(seq)
    win = [it.next() for cnt in xrange(windowSize)] # First window
    yield win
    for e in it: # Subsequent windows
        win[:-1] = win[1:]
        win[-1] = e
        yield win

def getMovingAverage(values, windowSize):
    movingAverages = []
    
    for w in rollingWindow(values, windowSize):
        movingAverages.append(sum(w)/len(w))

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

## ================================================================

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

## ================================================================

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

## ================================================================

def getTrainingData(stockSymbol):
    historicalData = getHistoricalData(stockSymbol)

    # reverse it so we're using the most recent data first, ensure we only have 9 data points
    historicalData.reverse()
    del historicalData[9:]

    # get five 5-day moving averages, 5-day lows, and 5-day highs, associated with the closing price
    trainingData = getTimeSeriesValues(historicalData, 5)

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

## ================================================================

def analyzeSymbol(stockSymbol):
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

    return returnData

## ================================================================

if __name__ == "__main__":
    print analyzeSymbol("GOOG")