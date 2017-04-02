"""
AJINKYA PADWAD
MARCH 2017
"""
import random

InputNodes = 3       # Input nodes including the bias node
MaxPatterns = 4      # Number of patterns for XOR
HiddenNodes = 4		# Number of hidden nodes in between
MaxEpoch = 1000		# Maximum batch runs

HiddenValue[HiddenNodes] = [0.0]            	# Initialise the hidden nodes
weightsIH[InputNodes][HiddenNodes] 				# Weights from input to hidden nodes
weightsHO[HiddenNodes] = [ 0.0 ]           		# Weights from hidden to output nodes
TrainInput[MaxPatterns][InputNodes]				# Input node values
TrainOutput[MaxPatterns]           				# Output obtained actually
RateLearningIH = 0.5       						# Learning rate for input to 
RateLearningHO= 0.05      						# Learning rate, hidden to output weights.
pattern = 0										# pattern iterations
CurrentError = 0.0								# error for current pattern intput
ExpectedOutput = 0.0                   			# Expected output values.
Error = 0.0                  					# Root Mean Squared error.
TargetError = 0.0




# function to generate random values for weights
def GetRandom():
	return random.uniform(0,1)

# function to set random weights 
def SetWeight():
	# Initialize weights to random values.
	print "\n	Initial Weights:-\n"
	for j in range(HiddenNodes):
		weightsHO[j] = (GetRandom() - 0.5) / 2	

		for i in range(InputNodes):
			weightsIH[i][j] = (GetRandom() - 0.5) / 5
			print "\t 	", weightsIH[i][j]


# funcion to initialise data at input nodes
def SetData():
	TrainInput[0][0] = 1
	TrainInput[0][1] = -1
	TrainInput[0][2] = 1 # Bias input
	TrainOutput[0] = 1

	TrainInput[1][0] = -1
	TrainInput[1][1] = 1
	TrainInput[1][2] = 1 # Bias input
	TrainOutput[1] = 1

	TrainInput[2][0] = 1
	TrainInput[2][1] = 1
	TrainInput[2][2] = 1 # Bias input
	TrainOutput[2] = -1

	TrainInput[3][0] = -1
	TrainInput[3][1] = -1
	TrainInput[3][2] = 1 # Bias input
	TrainOutput[3] = -1


# function to build ANN
def BuildNetwork():

	for i in range ( HiddenNodes ):
		HiddenValue[i] = 0.0

		for j in range( InputNodes ):			
			HiddenValue[i] = HiddenValue[i] + (TrainInput[pattern][j] * weightsIH[j][i])

		HiddenValue[i] = tanh(HiddenValue[i])

	ExpectedOutput = 0.0

	for i in range( HiddenNodes ):	
		ExpectedOutput = ExpectedOutput + HiddenValue[i] * weightsHO[i]
		
		#Calculate the error for the pattern as(Expected - Actual)
	CurrentError = ExpectedOutput - TrainOutput[pattern]


# functions to update the weights 
def UpdateWeightHO():
	
	for k in range(HiddenNodes):
		weightChange = RateLearningHO * CurrentError * HiddenValue[k]
		weightsHO[k] = weightsHO[k] - weightChange

		if (weightsHO[k] < -5):
			weightsHO[k] = -5

		elif (weightsHO[k] > 5):
			weightsHO[k] = 5

def UpdateWeightIH():
	for i in range(HiddenNodes ):
		for  k in range ( InputNodes):
			x = 1 - (HiddenValue[i] * HiddenValue[i])
			x = x * weightsHO[i] * CurrentError * RateLearningIH
			x = x * TrainInput[pattern][k]
			weightChange = x
			weightsIH[k][i] = weightsIH[k][i] - weightChange
	
# calculate RMS error overall
def FindError():
	Error = 0.0

	for i in range ( MaxPatterns ):
		pattern = i
		BuildNetwork()
		Error = Error + (CurrentError * CurrentError)
		
	Error = Error / MaxPatterns
	Error = sqrt(Error)

# function to display the output
def Display():
	
	print "\n"
	for i in range ( MaxPatterns ):
		pattern = i
		BuildNetwork()
		print pattern + 1 , ". "
		print " Expected  = ", TrainOutput[pattern]
		print " Predicted = ", ExpectedOutput 
	
# -------- MAIN PART -------------------------------------

TargetError = raw_input(" Enter Target Error:  ")
RateLearningIH = raw_input(" Enter Learning Rate:  ")
		
print "\n		Setting up weights .. "
SetWeight()

print "\n		Setting up data nodes .. "
SetData()

print "\n		Building the network now .."

# Training the network
for j in range(MaxEpoch ):
	
	for i in range( MaxPatterns ):

		pattern = random.uniform(0,1) % MaxPatterns		# pick a random pattern number
		BuildNetwork()					# generate the neural network
		UpdateWeightHO()					# get the changes in hidden to ouput weights
		UpdateWeightIH()					# get the changes in input to hidden weights

	FindError()		# calculate the error for the pattern
		
	if (j==0):
		print "\n Error - Batch I : ", Error 

	if (Error < TargetError):	# print output and stop once error < target error
		
		print "\n 	Final Weights:\n"

		for k in range( HiddenNodes ):
			for i in range( InputNodes ):
				print "\t 	",  weightsIH[i][k] 

		print "\n Error - Final Batch : ", Error, "\n\tin epoch number: ", j 
		break

# print the results 
Display()
print "\n Total number of epochs :",  MaxEpoch
