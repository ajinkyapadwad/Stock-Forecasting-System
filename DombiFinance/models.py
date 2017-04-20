# Create your models here.

from django.db import models
# from django.db import connection
# from yahoo_finance import Share
# import time 

# def HistoricalStocks():
# 	with connection.cursor() as cursor:
# 		for name in [Y,G,A,E,AP] :
# 			if name is Y:
# 				sql = """CREATE TABLE Yahoo ( date  CHAR(40),open FLOAT,high FLOAT,low FLOAT,price FLOAT,volume INT )"""
# 			if name is G:
# 				sql = """CREATE TABLE Google ( date  CHAR(40),open FLOAT,high FLOAT,low FLOAT,price FLOAT,volume INT )"""
# 			if name is A:
# 				sql = """CREATE TABLE Amazon ( date  CHAR(40),open FLOAT,high FLOAT,low FLOAT,price FLOAT,volume INT )"""
# 			if name is E:
# 				sql = """CREATE TABLE EASports ( date  CHAR(40),open FLOAT,high FLOAT,low FLOAT,price FLOAT,volume INT )"""
# 			if name is AP:
# 				sql = """CREATE TABLE Apple ( date  CHAR(40),open FLOAT,high FLOAT,low FLOAT,price FLOAT,volume INT )"""
		
# 			cursor.execute(sql)    # execute SQL query
				   
# 			dic = name.get_historical('2015-12-01', '2016-12-31')   # get historical data dictionary
			
# 			for i in range(len(dic)):   # scan dictionary data

# 				val=dic[i].values() # ignore keys, take just values 

# 				if name is Y:   # insert all above avlues into Yahoo table
# 					sq2 = "INSERT INTO Yahoo  (date, open, high, low, price, volume )VALUES ('%s', '%s', '%s', '%s', '%s', '%s' )" % (val[5], val[2], val[4], val[3], val[7], val[0])
# 				if name is G:   # insert all above avlues into Google table
# 					sq2 = "INSERT INTO Google (date, open, high, low, price, volume )VALUES ('%s', '%s', '%s', '%s', '%s', '%s' )" % (val[5], val[2], val[4], val[3], val[7], val[0])
# 				if name is A:      
# 					sq2 = "INSERT INTO Amazon  (date, open, high, low, price, volume )VALUES ('%s', '%s', '%s', '%s', '%s', '%s' )" % (val[5], val[2], val[4], val[3], val[7], val[0])
# 				if name is E:      
# 					sq2 = "INSERT INTO EASports  (date, open, high, low, price, volume )VALUES ('%s', '%s', '%s', '%s', '%s', '%s' )" % (val[5], val[2], val[4], val[3], val[7], val[0])
# 				if name is AP:      
# 					sq2 = "INSERT INTO Apple (date, open, high, low, price, volume )VALUES ('%s', '%s', '%s', '%s', '%s', '%s' )" % (val[5], val[2], val[4], val[3], val[7], val[0])
				
# 				cursor.execute(sq2)    # execute SQL query
		
# 			# cursor.execute("UPDATE bar SET foo = 1 WHERE baz = %s", [self.baz])
# 			# cursor.execute("SELECT foo FROM bar WHERE baz = %s", [self.baz])
# 			#row = cursor.fetchone()

# 	#

# HistoricalStocks()