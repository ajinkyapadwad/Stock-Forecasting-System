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

		for i in range ( HiddenNodes )
		
		HiddenValue[i] = 0.0

		for j in range( InputNodes )			
		HiddenValue[i] = HiddenValue[i] + (TrainInput[pattern][j] * weightsIH[j][i])

		HiddenValue[i] = tanh(HiddenValue[i])

		ExpectedOutput = 0.0

		for i in range( HiddenNodes )		
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

		else if (weightsHO[k] > 5):
			weightsHO[k] = 5

	def UpdateWeightIH():

		for i in range(HiddenNodes )
			for  k in range ( InputNodes)
				x = 1 - (HiddenValue[i] * HiddenValue[i])
				x = x * weightsHO[i] * CurrentError * RateLearningIH
				x = x * TrainInput[pattern][k]
				weightChange = x
				weightsIH[k][i] = weightsIH[k][i] - weightChange
	
	# calculate RMS error overall
	def FindError():
		Error = 0.0

		for i in range ( MaxPatterns )
			pattern = i
			BuildNetwork()
			Error = Error + (CurrentError * CurrentError)
		
		Error = Error / MaxPatterns
		Error = sqrt(Error)

	# function to display the output
	def Display()
	
		print "\n"
		for i in range ( MaxPatterns )
			pattern = i
			BuildNetwork()
			print pattern + 1 , ". "
			print " Expected  = ", TrainOutput[pattern]
			print " Predicted = ", ExpectedOutput 

	
# -------- MAIN PART -------------------------------------

	XOR_Network x	# instance for the class

	cout<<" 	Enter Target Error:  "
	cin >> TargetError
	cout<<"	Enter Learning Rate:  "
	cin >> RateLearningIH
	
	cout<<"\n		Setting up weights .. "<<endl
	x.SetWeight()

	cout<<"\n		Setting up data nodes .. "<<endl
	x.SetData()

	cout<<"\n		Building the network now .."<<endl

	# Training the network
	for (int j = 0 j <= MaxEpoch j++)
	
	for (int i = 0 i < MaxPatterns i++)

			pattern = rand() % MaxPatterns		# pick a random pattern number
			x.BuildNetwork()					# generate the neural network
			x.UpdateWeightHO()					# get the changes in hidden to ouput weights
			x.UpdateWeightIH()					# get the changes in input to hidden weights


		x.FindError()		# calculate the error for the pattern
		
		if (j==0)
		cout << "\n Error - Batch I : " <<Error << endl

		if (Error < TargetError)	# print output and stop once error < target error
		
		cout << "\n 	Final Weights:\n"
		for (int k = 0 k < HiddenNodes k++)

		for (int i = 0 i < InputNodes i++)


		cout << "\t 	" << weightsIH[i][k] << endl


		cout << "\n Error - Final Batch : "<< Error<<"\n\tin epoch number: "<< j <<endl
		break
		

	# print the results 
	x.Display()
	cout << "\n Total number of epochs :" << MaxEpoch<<endl

	return 0


