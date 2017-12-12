/* Nicholas Gao
 * ECE469 Artificial Intelligence
 * Fall 2017
 * Professor Sable
 * Project 2: Neural Network */

#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
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

net::layer *net::getLayer(int position) {
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
            // OUTPUT Layer
                auto outputLayer = this->getLayer(2);
                auto prevLayer = this->getLayer(1);
                double error;
                for(int on = 0; on < outputLayer->getNumNeurons(); on++) { // Each Output Node
                    auto outNode = outputLayer->getNeuron(on);

                    // Compute the Error at this Output Node
                        error = dsig(outNode->getInJ()) * (outputs[on]-(outNode->getOutput()));
                        outNode->setError(error);
                        cout << "O Error: "<<error<<endl;

                    // Update Weights coming into this node. Remember to update bias weight TOO.
                        // Bias Update
                        double newBiasW = outNode->getBiasWeight() + (learningRate*(-1)*error);
                        outNode->setBiasWeight(newBiasW);
                        // Update InWeights from Nodes (Prev Layer) -> This NODE
                        double newW;
                        for(int prevN = 0; prevN < prevLayer->getNumNeurons(); prevN++) {
                            auto prevNode = prevLayer->getNeuron(prevN);
                            newW = outNode->getWeight(prevN) + (learningRate*(prevNode->getOutput())*error);
                            outNode->setInWeight(newW,prevN);
                        }
                }
            // Other Layers (From l = L-1 to layer[1] (1st hidden layer)) [Don't need to compute error/weights for input layer]
            for(int l = 1; l > 0; l--) {
                auto layer = this->getLayer(l);
                auto prevLayer = this->getLayer(l-1);
                auto nextLayer = this->getLayer(l+1);
                double error;
                for(int n = 0; n < layer->getNumNeurons(); n++) { // Each node in Layer
                    auto node = layer->getNeuron(n);
                    error = 0;
                    // Compute the Error at this Node
                        for(int nextN = 0; nextN < nextLayer->getNumNeurons(); nextN++) { // Each node in NEXT Layer
                            auto nextNode = nextLayer->getNeuron(nextN);
                            // use n as index in next layer's node's inWeights
                            error += (nextNode->getWeight(n))*(nextNode->getError());
                        }
                        error *= dsig(node->getInJ());
                        cout << "In Error: "<<error<<endl;
                        node->setError(error);

                    // Update Weights coming into this node. Remember to update bias weight TOO.
                    // Bias Weight
                    double newBiasW = node->getBiasWeight() + (learningRate*(-1)*error);
                    node->setBiasWeight(newBiasW);
                    // Update InWeights from Nodes (Prev Layer) -> This NODE
                    double newW;
                    for(int prevN = 0; prevN < prevLayer->getNumNeurons(); prevN++) {
                        auto prevNode = prevLayer->getNeuron(prevN);
                        newW = node->getWeight(prevN) + (learningRate*(prevNode->getOutput())*error);
                        node->setInWeight(newW,prevN);
                    }
                }
            }

            // Training Example Done. Clear the in/out vectors for next example & free up memory
            inputs.clear();
            outputs.clear();
        }
    }
}

void net::saveNetwork(ofstream &saveFile) {
    int numI = this->getLayer(0)->getNumNeurons();
    int numH = this->getLayer(1)->getNumNeurons();
    int numO = this->getLayer(2)->getNumNeurons();
    saveFile << numI <<" "<< numH <<" "<< numO <<endl;

    net::layer *layer;
    // Output Hidden Layer Lines
        layer = this->getLayer(1);
        for(int hn = 0; hn < numH; hn++) {
            auto hNode = layer->getNeuron(hn);
            saveFile << hNode->getAllWeightsInString() << endl;
        }
    // Write Output Layer Lines
        layer = this->getLayer(2);
        for(int on = 0; on < numO; on++) {
            auto oNode = layer->getNeuron(on);
            saveFile << oNode->getAllWeightsInString() << endl;
        }
}