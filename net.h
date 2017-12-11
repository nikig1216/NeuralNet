/* Nicholas Gao
 * ECE469 Artificial Intelligence
 * Fall 2017
 * Professor Sable
 * Project 2: Neural Network */

#ifndef NEURALNET_NET_H
#define NEURALNET_NET_H

#include <vector>

using namespace std;

class net {
public:
    net();

    class neuron {
        public:
            neuron(int pos, double biasW/*, double output, double inW*/){
                this->position = pos;
                this->biasW = biasW;
//                this->output = output;
//                this->inWeights.push_back(inW);
            }
            void setBiasWeight(double value) {
                this->biasW = value;
            }
            void addInWeight(double inW) {
                this->inWeights.push_back(inW);
            }
            double getBiasWeight() {
                return this->biasW;
            }
            double getWeight(int pos) {
                return this->inWeights[pos];
            }
            void setInJ(double value) {
                this->inJ = value;
            }
            void setOutput(double output) {
                this->output = output;
            }
            double getOutput() {
                return this->output;
            }

        private:
            int position; // Index for neurons in a layer.

            double biasW;
            double inJ; // The sum of the linear combination of inputs
            double output; // The activation
            vector<double> inWeights; // Order follows order of nodes in prev layer

    };

    class layer {
        public:
            void addNewNeuron(int pos, double biasW) {
                auto n = new neuron(pos, biasW);
                this->neurons.push_back(n);
            }
            void addFinishedNeuron(neuron *n) {
                this->neurons.push_back(n);
            }
            neuron *getNeuron(int i) {
                return this->neurons[i];
            }
            int getNumNeurons() {
                return this->neurons.size();
            }
        private:
            vector<neuron *> neurons;
    };

    void addLayer();
    layer *getLayer(int position);

    void learn(ifstream &trainDataFile, int numEpochs, double learningRate);
    void saveNetwork(ofstream &saveFile);


private:

//    int numLayers;
    vector<layer *> layers;

};


#endif //NEURALNET_NET_H
