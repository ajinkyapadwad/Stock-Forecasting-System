from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse


def index(request):
    return HttpResponse("Hello, world. You're at the DombiFinance index.")

import MySQLdb                      #import SQL DB library
import datetime                     #import date time library
from yahoo_finance import Share     #import yahoo finance library
import time                         #import time library for delay

db = MySQLdb.connect("127.0.0.1","root","","Real-today" )    # DB 1 for real-time
db2 = MySQLdb.connect("127.0.0.1","root","","Historical" )  # DB for historical data

# cursors for the two DB's
cursor = db.cursor()    
cursor2 = db2.cursor()

# Ticker symbols for stocks - 
Y = Share('YHOO')
G = Share('GOOG')
A = Share('AMZN')
E = Share('EA')
AP = Share('AAPL')

def show()
    print " inside views.py "
    return render_to_response("blog.html",{})

# function to track real time data
def RealTimeStocks():

    # create five separate tables - 
    sql = """CREATE TABLE Yahoo(date CHAR(40),price FLOAT,volume INT )"""   # SQL query
    cursor.execute(sql)                                                     # execute SQL query
    sql = """CREATE TABLE Google(date CHAR(40),price FLOAT,volume INT )"""
    cursor.execute(sql)
    sql = """CREATE TABLE Amazon(date CHAR(40),price FLOAT,volume INT )"""
    cursor.execute(sql)
    sql = """CREATE TABLE EASports(date CHAR(40),price FLOAT,volume INT )"""
    cursor.execute(sql)
    sql = """CREATE TABLE Apple(date CHAR(40),price FLOAT,volume INT )"""
    cursor.execute(sql)

    # loop to track real time data
    for t in range(270):  # = must be set to 505 for tracking 7 hours (10 AM to 5 PM) @ 50 sec refresh rate
        for name in [Y,G,A,E,AP] :          # scan the five tickers
            date=datetime.datetime.now()   # get current system time
            price= name.get_price()         # get current stock price
            volume= name.get_volume()       # get current volume
            if name is Y:   # insert all above avlues into Yahoo table
                sq2 = "INSERT INTO Yahoo (date, price, volume )VALUES ('%s', '%s', '%s' )" % (date, price, volume)
            if name is G:   # insert into Google table
                sq2 = "INSERT INTO Google (date, price, volume )VALUES ('%s', '%s', '%s' )" % (date, price, volume)
            if name is A:
                sq2 = "INSERT INTO Amazon (date, price, volume )VALUES ('%s', '%s', '%s' )" % (date, price, volume)
            if name is E:
                sq2 = "INSERT INTO EASports (date, price, volume )VALUES ('%s', '%s', '%s' )" % (date, price, volume)
            if name is AP:
                sq2 = "INSERT INTO Apple (date, price, volume )VALUES ('%s', '%s', '%s' )" % (date, price, volume)
            cursor.execute(sq2) # execute SQL query
            db.commit()         # commit all the changes
        for name in [Y,G,A,E,AP] :
            time.sleep(15)   # = delay of 50 seconds expected. 
            name.refresh()  # refresh command

# function to log historical data
def HistoricalStocks():
    # create five separate tables -
    for name in [Y,G,A,E,AP] :
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
    
        cursor2.execute(sql)    # execute SQL query
               
        dic = name.get_historical('2015-12-01', '2016-12-31')   # get historical data dictionary
        
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
            
            cursor2.execute(sq2)    # execute SQL query

            db2.commit()    # commit all the changes

# Call the two functions
#RealTimeStocks()
HistoricalStocks()

# close databases
db.close()
db2.close()
