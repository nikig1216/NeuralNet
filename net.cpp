/* Nicholas Gao
 * ECE469 Artificial Intelligence
 * Fall 2017
 * Professor Sable
 * Project 2: Neural Network */

#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include "net.h"

using namespace std;

net::net() {
//    this->numLayers = numLayers;
}

double sig(double x) {
    return (1/(1+exp(-x)));
}
double dsig(double x) {
    double temp = sig(x);
    return (temp*(1-temp));
}

void net::addLayer() {
    auto l = new layer();
    this->layers.push_back(l);
}

layer *net::getLayer(int position) {
    return this->layers[position];
}

void net::learn(ifstream &trainDataFile, int numEpochs, double learningRate) {
    // Don't need to check that data has dimensionality of network.
    int numTrain, numI, numO;
    string line1;
    getline(trainDataFile, line1);
//        cout << "The first line: " << line1 << endl;
    stringstream ss(line1);
    ss >> numTrain;
    ss >> numI;
    ss >> numO;

    // Iterate through all training data (numEpochs) times.
    for(int numE = 0; numE < numEpochs; numE++) {
        // Iterate through each training example
        string line;
        for(int numTr = 0; numTr < numTrain; numTr++) {
            // Read in input and output data for a single training example
            vector<double> inputs;
            vector<double> outputs;

            getline(trainDataFile, line);
            stringstream ssTr(line);
            double value;
            for(int in = 0; in < numI; in++) {
                ssTr >> value;
                inputs.push_back(value);
            }
            for(int out = 0; out < numO; out++) {
                ssTr >> value;
                outputs.push_back(value);
            }

            /// Propagate inputs to outputs
            // Set inputs/Set inputs to output OR activation of input layer
            auto inputLayer = this->getLayer(0);
            for(int i = 0; i < numI; i++) {
                auto in = inputLayer->getNeuron(i);
                in->setOutput(inputs[i]);
            }
            // Set Activations OR Outputs of Other Layers (l from 1 to L=3) [Index 1 is the 2nd layer/1st Hidden Layer]
            for(int l = 1; l < 3; l++) {
                auto layer = this->getLayer(l);
                auto prevLayer = this->getLayer(l-1);
                for(int i = 0; i < layer->getNumNeurons(); i++) { // Each NODE in Layer
                    auto node = layer->getNeuron(i);
                    double inJ = 0;
                    inJ = -1*(node->getBiasWeight()); // Add Fixed Input

                    for(int h = 0; h < prevLayer->getNumNeurons(); h++) { // Add Inputs from EACH Prev Layer Node
                        inJ += (node->getWeight(h))*(prevLayer->getNeuron(h)->getOutput());
                    }
                    node->setInJ(inJ);

                    double a = sig(inJ);
                    node->setOutput(a);
                }
            }

            /// Back-Propagate
            vector<double> errors;

            // Training Example Done. Clear the in/out vectors for next example & free up memory
            inputs.clear();
            outputs.clear();
        }
    }
}

void net::saveNetwork(ofstream &saveFile) {

}