/* Nicholas Gao
 * ECE469 Artificial Intelligence
 * Fall 2017
 * Professor Sable
 * Project 2: Neural Network */

#include <iostream>
#include <fstream>
#include <sstream>
#include "net.h"

void startTrain() ;
void startTest() ;

using namespace std;

void getPosInteger(string message, int &ref) {
    bool inputGood = false;
    while (!inputGood) {
        inputGood = true;

        cout << message;
        cin >> ref;

        if (!cin || ref <= 0) {
            // Non-integer in input buffer, get out of "fail" state
            cin.clear();
            inputGood = false;
        }
        while (cin.get() != '\n'); // clear buffer
    }
}
void getDouble(string message, double &ref) {
    bool inputGood = false;
    while (!inputGood) {
        inputGood = true;

        cout << message;
        cin >> ref;

        if (!cin) {
            // Non-integer in input buffer, get out of "fail" state
            cin.clear();
            inputGood = false;
        }
        while (cin.get() != '\n'); // clear buffer
    }
}

int main() {
    cout<<"Welcome to my neural net program!"<<endl;
    char mode;
    while(mode != '1' && mode != '2') {
        cout << "Choose a mode number: 1) TRAIN a Neural Network OR 2) TEST a Neural Network" << endl;
        cin >> mode;
        cin.clear();
        while (cin.get() != '\n');
    }
    if(mode == '1') {
        startTrain();
    }
    else {
        startTest();
    }
    return 0;
}

void startTrain() {
    string inFile, trainFile, outFile;
    int numEp;
    double rate;
    /// Ask for input file, training set, and output file
    cout<<"Please enter the input file's name containing the specified initial neural network: ";
    cin >> inFile;
    cin.clear();
    cin.ignore(10000, '\n');

    cout<<"Please enter the training set file's name: ";
    cin >> trainFile;
    cin.clear();
    cin.ignore(10000, '\n');

    cout<<"Please enter the desired output file's name to store the trained network: ";
    cin >> outFile;
    cin.clear();
    cin.ignore(10000, '\n');

    /// Ask for number of epochs and training rate
    getPosInteger("Please enter a positive integer value for the number of epochs: ", numEp);
    getDouble("Please enter a floating-point value for the training rate: ", rate);

    /// BUILD the Neural Network
    ifstream networkFileStream;
    networkFileStream.open(inFile.c_str());

    int numI, numH, numO;

    net Network;

    if(networkFileStream.is_open()) {
        string line1;
        getline(networkFileStream, line1);
//        cout << "The first line: " << line1 << endl;
        stringstream ss(line1);
        ss >> numI;
        ss >> numH;
        ss >> numO;
//        cout << "numIn: " << numI << " | numHidden: " << numH << " | numOut: " << numO << endl;

        // Generate layers and neurons.
        // Input Nodes
            Network.addLayer();
            auto inputLayer = Network.getLayer(0);
            for(int n = 0; n < numI; n++) {
                inputLayer->addNewNeuron(n, 0); // Output is set when running the network
                // No inWeights for input layer. This is considered the first layer, or layer[0]
            }

        // Read in the next numH lines. first number is bias of hidden node. add the respective weight in each row to each of the input nodes.
            Network.addLayer();
            auto hiddenLayer = Network.getLayer(1);
            string HLine;

            for(int i = 0; i < numH; i++) {
                getline(networkFileStream, HLine);
                stringstream ssH(HLine);

                // Read the Bias Weight
                double biasWeight;
                ssH >> biasWeight;
                auto hn = new net::neuron(i,biasWeight);
                // Read the rest of the line.
                double inWeight;
                for(int j = 0; j < numI; j++) {
                    ssH >> inWeight;
                    hn->addInWeight(inWeight); // Output set later, during run.
                }
                hiddenLayer->addFinishedNeuron(hn);
            }

        // Read in the next numO lines. first number is bias of output node. add the respective weight in each row to each of the hidden nodes.
            Network.addLayer();
            auto outputLayer = Network.getLayer(2);
            string OLine;

            for(int i = 0; i < numO; i++) {
                getline(networkFileStream,OLine);
                stringstream ssO(OLine);

                // Read the Bias Weight
                double biasWeight;
                ssO >> biasWeight;
                auto on = new net::neuron(i,biasWeight);

                // Read the rest of weights/rest of line
                double inWeight;
                for(int j = 0; j < numH; j++) {
                    ssO >> inWeight;
                    on->addInWeight(inWeight); // Output is set later
                }
                outputLayer->addFinishedNeuron(on);
            }
        // Done Building
        networkFileStream.close();
    }
    else {
        cerr << "ERROR: Cannot open the initial network file. Program terminating." <<endl;
        return;
    }

    /// TRAINING
        ifstream trainFileStream;
        trainFileStream.open(trainFile.c_str());

        if(trainFileStream.is_open()) {
            Network.learn(trainFileStream, numEp, rate);
            trainFileStream.close();
        }
        else {
            cerr << "ERROR: Cannot open the training data file. Program terminating." <<endl;
            return;
        }

    /// OUTPUT/SAVE Trained Network
//        ofstream saveFileStream;
//        saveFileStream.open(outFile.c_str());
//
//        Network.saveNetwork(saveFileStream);

    return;
}

void startTest() {
    return;
}