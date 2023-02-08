#include "neural_net.cpp"
#include <iostream>
#include <random>

using namespace std;


// generate a dataset for the XOR problem with noise and extra columns according to parameters. The default is a clean dataset. The first two columns are the XOR input
LabeledData generate_XOR_test_data(int noise_magnitude=0, int n_meaningless_columns=0)
{
    // create a random number generator for extra meaningless columns
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    // a rng from 0 to 3 for random selection of the xor input
    std::uniform_int_distribution<> dis2(0, 3);

    // a rng from 0 to 1 of doubles with 3 digits of precision for noise
    std::uniform_real_distribution<> dis3(0.0, 1.0);


    int xor_labels[4] = {0, 1, 1, 0};
    int xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    // create the data
    int n_data_points = 4 + n_meaningless_columns*2;    // generate more data when adding meaningless columns
    int n_features = 2 + n_meaningless_columns;
    double** features = new double*[n_data_points];
    double** labels = new double*[n_data_points];

    // basic without extra columns
    if (n_meaningless_columns == 0){
        for (int i=0; i<4; i++){
            features[i] = new double[2];
            for (int j=0; j<2; j++){
                features[i][j] = xor_inputs[i][j];
            }
            labels[i] = new double[1];
            labels[i][0] = xor_labels[i];
        }
    // with extra columns
    }else{
        for (int i=0; i<n_features; i++){
            features[i] = new double[2 + n_meaningless_columns];
            // get a random xor input for the first 2 columns and label
            int index = dis2(gen);
            for (int j=0; j<2; j++){
                features[i][j] = xor_inputs[index][j];
            }
            labels[i][0] = xor_labels[index];
            // add meaningless columns
            for (int j=2; j<n_features; j++){
                features[i][j] = dis(gen);
            }
        }
    }

    if (noise_magnitude == 0)
    {
        return LabeledData{features, labels, n_data_points};
    }

    // add noise to the data
    for (int i=0; i<n_data_points; i++){
        for (int j=0; j<n_features; j++){
            features[i][j] += dis3(gen)*noise_magnitude;
        }
    }
    printf("Data has n data points: %d\n", n_data_points);
    return LabeledData{features, labels, n_data_points};
}


int main(){
    auto xor_data = generate_XOR_test_data();
    printf("XOR training data:\n");
    print_2D_array(xor_data.features, 4, 2);

    printf("XOR training labels:\n");
    print_2D_array(xor_data.labels, 4, 1);

    // create an instance of the neural network (initializes with random weights)
    // define the neural net dimensions: input, hidden nodes per layer, hidden layers, output nodes
    Hyperparameters hp = DEFAULT_HYPERPARAMS;
    hp.momentum = 0;
    hp.learning_rate = 1;
    NeuralNet neural_net(2, 1, 4, 1, hp);

    // train the neural network, recording the scores
    int n_epochs = 20;
    
    Vector losses = neural_net.train(xor_data, n_epochs);    // note there is an implicit accuracy threshold of 0.99

    // starting loss
    printf("Starting loss: %f\n", losses.array[0]);
    // ending loss
    printf("Ending loss: %f\n", losses.array[n_epochs-1]);
    
    printf("Final weights and biases:\n");
    neural_net.print_weights_and_biases();

    neural_net.save_to_file();
    losses.save_to_file("losses.csv");
}