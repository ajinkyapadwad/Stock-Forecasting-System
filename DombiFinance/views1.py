from django.http import HttpResponse
from django.shortcuts import render
import MySQLdb
import numpy
import math

db = MySQLdb.connect("127.0.0.1","root","","Real-Time" )    # DB 1 for real-time
db2 = MySQLdb.connect("127.0.0.1","root","","Historical" )  # DB for historical data

# cursors for the two DB's
cursor = db.cursor()    
cursor2 = db2.cursor()


def bayesian(data):
    x_10 =[]
    t_data = []
    for i in xrange(len(data) - 90, len(data)):
        t_data.append(data[i])
    for i in xrange(1, 11):
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
        #print "Predicted Price = bayesian mean :" , mean

    t = t_data[0]
    t_data = t
    sum = 0
    avg = 0
    for i in t_data:
        sum += i
    avg = sum / len(t_data)
    print 'mov', avg
    per = ((mean - avg) / avg) * 100
    print 'error % = ', per
    final = []
    mean = round(mean, 3)
    per = round(per, 3)
    final.append(mean)
    final.append(per)
    return final

# Prepare SQL query to INSERT a record into the database.
data=[]
sql = "SELECT price FROM Yahoo"
cursor.execute(sql)
results = cursor.fetchall()
for row in results:
      data.append(row[0])
print 'price', results
# data1 = [21.45,23.45,21.34,22.32,23.18]
# data2 = [100.12,101.64,102.20,99.65,100.42]#ADP
# data3 = [14.93,14.75,15.05,15.50,15.45]#NYT
# data4 = [825.64,831.33,830.76,831.66,828.07]#Google
# data5 = [135.18,135.36,136.12,133.12,133.53]#FB
# data6 = [28.32,28.50,27.91,27.37,28.26,28.55,28.65,29.05,28.64,28.11]
# data7 = [25.67,26.87,28.55,29.32,28.26,28.55,30.18,32.11,29.14,28.11]
# data8 = [125.67,126.87,128.55,132.44,123.55,128.88,130.12,134.5,139.21,137.45]
# data9 = [325.67,331.87,331.55,330.42,333.55,332.88,330.12,334.5,335.21,334.45]
# data10 = [1325.67,1321.87,1331.55,1334.42,1333.15,1328.88,1324.12,1330.35,1335.21,1334.45]
prediction = bayesian(data)
print 'Data'
# print 'data1 = [21.45,23.45,21.34,22.32,23.18]'
print 'Final Prediction', prediction
# print 'Data2'
# prediction = bayesian(data2)
# print 'data2 = [100.12,101.64,102.20,99.65,100.42]'
# print 'Final Prediction', prediction
# print 'Data3'
# print 'data3 = [14.93,14.75,15.05,15.50,15.45]'
# prediction = bayesian(data3)
# print 'Final Prediction', prediction
# print 'Data4'
# print 'data4 = [825.64,831.33,830.76,831.66,828.07]'
# prediction = bayesian(data4)
# print 'Final Prediction', prediction
# print 'Data5'
# print ''
# prediction = bayesian(data5)
# print 'data5 = [135.18,135.36,136.12,133.12,133.53]'
# print 'Final Prediction', prediction
# print 'Data6'
# prediction = bayesian(data6)
# print 'data6 = [28.32,28.50,27.91,27.37,28.26,28.55,28.65,29.05,28.64,28.11]'
# print 'Final Prediction', prediction
# print 'Data7'
# prediction = bayesian(data7)
# print 'data7 = [25.67,26.87,28.55,29.32,28.26,28.55,30.18,32.11,29.14,28.11]'
# print 'Final Prediction', prediction
# print 'Data8'
# print 'data8 = [125.67,126.87,128.55,132.44,123.55,128.88,130.12,134.5,139.21,137.45]'
# prediction = bayesian(data8)
# print 'Final Prediction', prediction
# print 'Data9'
# print 'data9 = [325.67,331.87,331.55,330.42,333.55,332.88,330.12,334.5,335.21,334.45]'
# prediction = bayesian(data9)
# print 'Final Prediction', prediction
# print 'Data10'
# print 'data10 = [1325.67,1321.87,1331.55,1334.42,1333.15,1328.88,1324.12,1330.35,1335.21,1334.45]'
# prediction = bayesian(data10)
# print 'Final Prediction', prediction
def index(request):
   return render(request,'index.html')