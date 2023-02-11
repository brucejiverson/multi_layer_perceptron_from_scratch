// initializes the weights and biases found in the "test_model" folder, which match the calculations on this website: 
// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/


#include "neural_net.cpp"
#include <iostream>

int main(){
    // create an instance of the neural network (initializes with random weights)
    // define the neural net dimensions: input, hidden nodes per layer, hidden layers, output nodes

    double** features = new double*[1];
    features[0] = new double[2];
    features[0][0] = 0.05; 
    features[0][1] = 0.1;

    double** labels = new double*[1];
    labels[0] = new double[1];
    labels[0][0] = 0.01;
    labels[0][1] = 0.99;

    LabeledData data = {Matrix(features, 1, 2), Matrix(labels, 1, 2)}; // inputs, outputs, n samples

    // train the neural network, recording the scores
    NeuralNet neural_net(2, 2, 2);
    neural_net.hyper_params.learning_rate = 0.5;
    neural_net.hyper_params.momentum = 0;
    int n_epochs = 1;

    neural_net.load_from_file("test_model");
    
    Vector losses = neural_net.train(data, n_epochs);
    printf("Final weights and biases:\n");
    neural_net.print_weights_and_biases();

    printf("Starting loss: %f\n", losses.array[0]);
    printf("Ending loss: %f\n", losses.array[n_epochs-1]);

    Vector last_loss = neural_net.train(data, n_epochs);
    printf("last loss: %f\n", last_loss.array[0]);
    
}

