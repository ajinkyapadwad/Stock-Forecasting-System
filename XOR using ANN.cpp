/*
NAME : AJINKYA PADWAD [aap239]
DATE : MARCH 24 2017
Artificial Neural Network - XOR function 
*/

#include <iostream>
#include <cstdlib>
#include <math.h>

using namespace std;

const int InputNodes = 3;       // Input nodes including the bias node
const int MaxPatterns = 4;      // Number of patterns for XOR
const int HiddenNodes = 4;		// Number of hidden nodes in between
const int MaxEpoch = 1000;		// Maximum batch runs

double HiddenValue[HiddenNodes] = { 0.0 };           	// Initialise the hidden nodes
double weightsIH[InputNodes][HiddenNodes]; 				// Weights from input to hidden nodes
double weightsHO[HiddenNodes] = { 0.0 };           		// Weights from hidden to output nodes
int TrainInput[MaxPatterns][InputNodes];				// Input node values
int TrainOutput[MaxPatterns];           				// Output obtained actually
double RateLearningIH = 0.5;       						// Learning rate for input to 
double RateLearningHO= 0.05;      						// Learning rate, hidden to output weights.
int pattern = 0;										// pattern iterations
double CurrentError = 0.0;								// error for current pattern intput
double ExpectedOutput = 0.0;                   			// Expected output values.
double Error = 0.0;                  					// Root Mean Squared error.
double TargetError = 0.0;

// class for building ANN for XOR operation
class XOR_Network
{
public:

	// function to generate random values for weights
	double GetRandom()
	{
		return double(rand() / double(RAND_MAX));
	}

	// function to set random weights 
	void SetWeight()
	{
		// Initialize weights to random values.
		cout << "\n	Initial Weights:-\n";
		for (int j = 0; j < HiddenNodes; j++)
		{

			weightsHO[j] = (GetRandom() - 0.5) / 2;
			for (int i = 0; i < InputNodes; i++){

				weightsIH[i][j] = (GetRandom() - 0.5) / 5;
				cout << "\t 	"<< weightsIH[i][j] << endl;
			}
		}
	}

	// funcion to initialise data at input nodes
	void SetData()
	{
		TrainInput[0][0] = 1;
		TrainInput[0][1] = -1;
		TrainInput[0][2] = 1; // Bias input
		TrainOutput[0] = 1;

		TrainInput[1][0] = -1;
		TrainInput[1][1] = 1;
		TrainInput[1][2] = 1; // Bias input
		TrainOutput[1] = 1;

		TrainInput[2][0] = 1;
		TrainInput[2][1] = 1;
		TrainInput[2][2] = 1; // Bias input
		TrainOutput[2] = -1;

		TrainInput[3][0] = -1;
		TrainInput[3][1] = -1;
		TrainInput[3][2] = 1; // Bias input
		TrainOutput[3] = -1;
	}

	// function to build ANN
	void BuildNetwork()
	{
		for (int i = 0; i < HiddenNodes; i++)
		{
			HiddenValue[i] = 0.0;

			for (int j = 0; j < InputNodes; j++)
			{
				HiddenValue[i] = HiddenValue[i] + (TrainInput[pattern][j] * weightsIH[j][i]);
			}

			HiddenValue[i] = tanh(HiddenValue[i]);
			
		}

		ExpectedOutput = 0.0;

		for (int i = 0; i < HiddenNodes; i++)
		{
			ExpectedOutput = ExpectedOutput + HiddenValue[i] * weightsHO[i];
		}
		//Calculate the error for the pattern as(Expected - Actual)
		CurrentError = ExpectedOutput - TrainOutput[pattern];
	}

	// functions to update the weights 
	void UpdateWeightHO()
	{
		for (int k = 0; k < HiddenNodes; k++)
		{
			double weightChange = RateLearningHO * CurrentError * HiddenValue[k];
			weightsHO[k] = weightsHO[k] - weightChange;

			if (weightsHO[k] < -5){
				weightsHO[k] = -5;
			}
			else if (weightsHO[k] > 5){
				weightsHO[k] = 5;
			}
		}
	}

	void UpdateWeightIH()
	{
		for (int i = 0; i < HiddenNodes; i++){

			for (int k = 0; k < InputNodes; k++){

				double x = 1 - (HiddenValue[i] * HiddenValue[i]);
				x = x * weightsHO[i] * CurrentError * RateLearningIH;
				x = x * TrainInput[pattern][k];
				double weightChange = x;
				weightsIH[k][i] = weightsIH[k][i] - weightChange;
			}
		}
	}

	// calculate RMS error overall
	void FindError()
	{  
		Error = 0.0;

		for (int i = 0; i < MaxPatterns; i++){
			pattern = i;
			BuildNetwork();
			Error = Error + (CurrentError * CurrentError);
		}

		Error = Error / MaxPatterns;
		Error = sqrt(Error);
	}

	// function to display the output
	void Display()
	{
		cout << "\n";
		for (int i = 0; i < MaxPatterns; i++){
			pattern = i;
			BuildNetwork();
			cout << pattern + 1 <<". "<<
			" Expected  = " << TrainOutput[pattern] <<
			" Predicted = " << ExpectedOutput << endl;
		}
	}
};

// main function
int main()
{
	XOR_Network x;	// instance for the class

	cout<<" 	Enter Target Error:  ";
	cin >> TargetError;
	cout<<"	Enter Learning Rate:  ";
	cin >> RateLearningIH;
	
	cout<<"\n		Setting up weights .. "<<endl;
	x.SetWeight();

	cout<<"\n		Setting up data nodes .. "<<endl;
	x.SetData();

	cout<<"\n		Building the network now .."<<endl;

	// Training the network
	for (int j = 0; j <= MaxEpoch; j++)
	{
		for (int i = 0; i < MaxPatterns; i++)
		{
			pattern = rand() % MaxPatterns;		// pick a random pattern number
			x.BuildNetwork();					// generate the neural network
			x.UpdateWeightHO();					// get the changes in hidden to ouput weights
			x.UpdateWeightIH();					// get the changes in input to hidden weights
		}

		x.FindError();		// calculate the error for the pattern
		
		if (j==0)
			cout << "\n Error - Batch I : " <<Error << endl;

		if (Error < TargetError)	// print output and stop once error < target error
		{
			cout << "\n 	Final Weights:\n";
			for (int k = 0; k < HiddenNodes; k++)
			{
				for (int i = 0; i < InputNodes; i++)
				{

					cout << "\t 	" << weightsIH[i][k] << endl;
				}
			}
			cout << "\n Error - Final Batch : "<< Error<<"\n\tin epoch number: "<< j <<endl;
			break;
		}
	}
	// print the results 
	x.Display();
	cout << "\n Total number of epochs :" << MaxEpoch<<endl;

	return 0;
}

