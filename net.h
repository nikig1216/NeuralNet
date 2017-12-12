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

    class metrics {
    public:
        class ContTable {
        public:
            ContTable() {
                this->A = 0;
                this->B = 0;
                this->C = 0;
                this->D = 0;
            }
            void findOtherMetrics() {
                // Cast ints to doubles
                double a = this->A;
                double b = this->B;
                double c = this->C;
                double d = this->D;

                this->OA = ((a+d)/(a+b+c+d));
                this->P = (a/(a+b));
                this->R = (a/(a+c));
                this->F1 = ((2*this->P*this->R)/(this->P+this->R));
            }

            int A;
            int B;
            int C;
            int D;
            double OA;
            double P;
            double R;
            double F1;
        };

        metrics(int numOutNodes) {
            for(int n = 0; n < numOutNodes; n++) {
                this->addContTable();
            }
        }

        void addContTable() {
            auto CT = new ContTable();
            this->tables.push_back(CT);
        }

        void getAllOtherMetricsForeachTable() {
            for(int i = 0; i < this->tables.size(); i++) {
                this->tables[i]->findOtherMetrics();
            }
        }

        void getOverallMetrics() {
            double numClass = this->tables.size();
            // Micro-averaging
            double tA = 0, tB = 0, tC = 0, tD = 0;
            for(int i = 0; i < numClass; i++) {
                tA += this->tables[i]->A;
                tB += this->tables[i]->B;
                tC += this->tables[i]->C;
                tD += this->tables[i]->D;
            }
            this->m_OA = ((tA+tD)/(tA+tB+tC+tD));
            this->m_P = (tA/(tA+tB));
            this->m_R = (tA/(tA+tC));
            this->m_F1 = ((2*this->m_P*this->m_R)/(this->m_P+this->m_R));

            // Macro-averaging
            double tOA = 0, tP = 0, tR = 0;
            for(int i = 0; i < numClass; i++) {
                tOA += this->tables[i]->OA;
                tP += this->tables[i]->P;
                tR += this->tables[i]->R;
            }
            this->M_OA = tOA/numClass;
            this->M_P = tP/numClass;
            this->M_R = tR/numClass;
            this->M_F1 = ((2*this->M_P*this->M_R)/(this->M_P+this->M_R));
        }

        void save(ofstream &saveFile) {
            // Output line for each output node | 8 values
            for(int o = 0; o < this->tables.size(); o++) {
                stringstream ss;
                auto t = this->tables[o];
                ss << t->A <<" "<<t->B<<" "<<t->C<<" "<<t->D<<" "<< std::setprecision(3) << std::fixed << t->OA << " "<< t->P<<" "<<t->R<<" "<<t->F1<<endl;
                saveFile << ss.str();
            }
            // Output the 2 Lines for Micro and Macro-Averaged Metrics
            stringstream ssm;
            ssm << std::setprecision(3) << std::fixed << this->m_OA << " "<< this->m_P<<" "<<this->m_R<<" "<<this->m_F1<<endl;
            saveFile << ssm.str();
            stringstream ssM;
            ssM << std::setprecision(3) << std::fixed << this->M_OA << " "<< this->M_P<<" "<<this->M_R<<" "<<this->M_F1<<endl;
            saveFile << ssM.str();

        }

        vector<ContTable *> tables;
        // M (Macro-averaging) | m (Micro-averaging)
        double m_OA;
        double m_P;
        double m_R;
        double m_F1;
        double M_OA;
        double M_P;
        double M_R;
        double M_F1;

    };

    void addLayer();
    layer *getLayer(int position);

    void learn(ifstream &trainDataFile, int numEpochs, double learningRate);
    void saveNetwork(ofstream &saveFile);

    void test(ifstream &testDataFile, ofstream &saveFile);


private:

//    int numLayers;
    vector<layer *> layers;

};


#endif //NEURALNET_NET_H
