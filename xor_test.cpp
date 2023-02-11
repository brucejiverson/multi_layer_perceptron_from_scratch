#include "neural_net.cpp"
#include <iostream>
#include <random>

using namespace std;

// pass by reference
void add_noise_to_array(double** arr, int n_rows, int n_cols, double noise_magnitude){
    // create a random number generator for extra meaningless columns
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    // a rng from 0 to 3 for random selection of the xor input
    std::uniform_int_distribution<> dis2(0, 3);

    // a rng from 0 to 1 of doubles with 3 digits of precision for noise
    std::uniform_real_distribution<> dis3(0.0, 1.0);

    for (int i=0; i<n_rows/2; i++){
        for (int j=0; j<n_cols; j++){
            arr[i][j] = arr[i][j] + (dis3(gen)- 0.5)*2*noise_magnitude;
        }
    }
}


// generate a dataset for the XOR problem with noise and extra columns according to parameters. The default is a clean dataset. The first two columns are the XOR input
LabeledData generate_XOR_test_data(int noise_magnitude=0)
{
    int xor_labels[4] = {0, 1, 1, 0};
    int xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    // create the data
    int n_data_points = 4;
    int n_features = 2;
    double** features = new double*[n_data_points];
    double** labels = new double*[n_data_points];

    for (int i=0; i<4; i++){
        features[i] = new double[2];
        for (int j=0; j<2; j++){
            features[i][j] = xor_inputs[i][j];
        }
        labels[i] = new double[1];
        labels[i][0] = xor_labels[i];
    }

    if (noise_magnitude == 0){
        return LabeledData{
            Matrix(features, n_data_points, 2), 
            Matrix(labels, n_data_points, 1)
            };
    }

    // create new arrays that are the previous array values but copied 4 times
    n_data_points *= 4;
    double** new_features = new double*[n_data_points];
    double** new_labels = new double*[n_data_points];
    for (int i=0; i<n_data_points; i++){
        new_features[i] = new double[2];
        new_labels[i] = new double[1];
        for (int j=0; j<2; j++){
            new_features[i][j] = features[i%4][j];
        }
        new_labels[i][0] = labels[i%4][0];
    }
    // add noise to the expanded samples
    add_noise_to_array(new_features, n_data_points, 2, noise_magnitude);

    printf("Data has n data points: %d\n", n_data_points);
    return LabeledData{
        Matrix(new_features, n_data_points, 2),
        Matrix(new_labels, n_data_points, 1)
    };
}

 
int main(){
    auto data = generate_XOR_test_data(0.15);
    data.features.print("XOR training data");
    data.labels.print("XOR training labels");

    NeuralNet neural_net(data.features.n_cols, data.labels.n_cols, 10, 2);
    neural_net.hyper_params.learning_rate = 0.5;
    neural_net.hyper_params.momentum = 0;
    int n_epochs = 10000;
    
    Vector losses = neural_net.train(data, n_epochs);
    printf("Final weights and biases:\n");
    neural_net.print_weights_and_biases();

    printf("Starting loss: %f\n", losses.array[0]);
    printf("Ending loss: %f\n", losses.array[n_epochs-1]);

    neural_net.save_to_file();
    losses.save_to_file("training_records/losses.csv");
}

