//
// Created by Nicholas Gao on 12/6/17.
//

#ifndef NEURALNET_NET_H
#define NEURALNET_NET_H

#include <vector>

using namespace std;

class net {
public:
    net(int numLayers);

    class neuron {
        public:
            neuron(int pos, double bias, double output, double outW){
                this->position = pos;
                this->bias = bias;
                this->outputs.push_back(output);
                this->outWeights.push_back(outW);
            }

        private:
            int position; // Index for neurons in a layer.

            double bias;
            vector<double> outputs;
            vector<double> outWeights;

    };

    class layer {
        public:
            void addNeuron(int pos, double bias, double output, double outW) {
                auto n = new neuron(pos, bias, output, outW);
                this->neurons.push_back(n);
            };
        private:
            vector<neuron *> neurons;
        };

    void addLayer();


private:
    void learn();

    int numLayers;
    vector<layer *> layers;
};


#endif //NEURALNET_NET_H
