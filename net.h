/* Nicholas Gao
 * ECE469 Artificial Intelligence
 * Fall 2017
 * Professor Sable
 * Project 2: Neural Network */

#ifndef NEURALNET_NET_H
#define NEURALNET_NET_H

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

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
            double getBiasWeight() {
                return this->biasW;
            }
            void addInWeight(double inW) {
                this->inWeights.push_back(inW);
            }
            void setInWeight(double value, int weightPos) {
                this->inWeights[weightPos] = value;
            }
            double getWeight(int pos) {
                return this->inWeights[pos];
            }
            string getAllWeightsInString() {
                stringstream ss;
                ss << std::setprecision(3) << std::fixed << this->getBiasWeight();
//                string output ( to_string(this->getBiasWeight()) );
                for(std::vector<double>::iterator it = this->inWeights.begin(); it != this->inWeights.end(); ++it) {
//                    output += " " + to_string(*it);
                    ss << std::setprecision(3) << std::fixed << " " << *it;
                }
//                return output;
                return ss.str();
            }
            void setInJ(double value) {
                this->inJ = value;
            }
            double getInJ() {
                return this->inJ;
            }
            void setOutput(double output) {
                this->output = output;
            }
            double getOutput() {
                return this->output;
            }
            void setError(double value) {
                this->error = value;
            }
            double getError() {
                return this->error;
            }

        private:
            int position; // Index for neurons in a layer.

            double biasW;
            double inJ; // The sum of the linear combination of inputs
            double output; // The activation
            vector<double> inWeights; // Order follows order of nodes in prev layer
            double error;

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
