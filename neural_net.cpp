#include<iostream>
#include<random>

#include"linear_algebra.cpp"
#include"machine_learning_helpers.cpp"
using namespace std;


double reverse_logistic(double x){
    // the inverse of the logistic function
    return log(x/(1-x));
}


double** get_2D_array_subset(double** X, int start_row, int end_row){
    // get a subset of a 2D array
    // start_row and end_row are inclusive
    int n_rows = end_row - start_row + 1;
    double** subset = new double*[n_rows];
    for (int i=0; i<n_rows; i++){
        subset[i] = X[start_row + i];
    }
    return subset;
}


struct Hyperparameters
{
    double learning_rate;
    double momentum;
    int minibatch_size = 32;    // the number of samples to do each round of backpropogation with
};


const Hyperparameters DEFAULT_HYPERPARAMS = {0.1, 0.9};


// a struct to hold the metadata for the feedforward pass for back prop
struct FeedForwardData{
    Vector* layer_outputs;
    Vector output;
};


// the information for backpropogation pass. loss, delta weights in, delta W hidden, delta W out, delta bias hidden, delta bias out
struct BackPropData{
    double loss=0;
    Matrix dWInput;
    Matrix* dWHidden;
    Matrix dWOutput;
    Matrix dBHidden;
    Vector dBOutput;
    int n_hidden_layers;
};


BackPropData get_copy(BackPropData data){
    // get a copy of the back prop data
    Matrix* dWHidden_copy; 
    for (int i=0; i<data.n_hidden_layers-1; i++){
        dWHidden_copy = new Matrix[data.n_hidden_layers-1];
        dWHidden_copy[i] = data.dWHidden[i].copy();
    }
    BackPropData copy = {
        data.loss, 
        data.dWInput.copy(), 
        dWHidden_copy,
        data.dWOutput.copy(), 
        data.dBHidden.copy(), 
        data.dBOutput.copy(),
        data.n_hidden_layers
        };
    return copy;
}


// scales everything except for the loss
BackPropData get_scaled_copy_of_backpropdata(BackPropData data, double scale_factor){
    // scale all the weights and biases in the back prop data by a scale factor
    BackPropData copy = get_copy(data);
    copy.dWInput.scale(scale_factor);
    copy.dWOutput.scale(scale_factor);
    copy.dBHidden.scale(scale_factor);
    copy.dBOutput.scale(scale_factor);
    for (int i=0; i<copy.n_hidden_layers - 1; i++){
        copy.dWHidden->scale(scale_factor);
    }
    return copy;
}


// adds the weights and biases in data2 to data1
void back_prop_weights_and_biases_plus_equals(BackPropData data1, BackPropData data2){
    matrix_plus_equal(data1.dWInput, data2.dWInput);
    matrix_plus_equal(data1.dWOutput, data2.dWOutput);
    matrix_plus_equal(data1.dBHidden, data2.dBHidden);
    vector_plus_equal(data1.dBOutput, data2.dBOutput);
    for (int i=0; i<data1.n_hidden_layers-1; i++){
        matrix_plus_equal(data1.dWHidden[i], data2.dWHidden[i]);
    }
}


// a class representing a dense (fully connected) neural network able to handle arbitrary sizes of input, hidden, and output layers and arbitrary # of hidden layers.
// Weights are initialized randomly and biases are initialized to 0
class NeuralNet
{
    public:
        int n_input_nodes;
        int n_hidden_nodes;
        int n_hidden_layers = 1;
        int n_output_nodes;
        Hyperparameters hyper_params;

        // constructor for neural net parameterse and hyperparameters
        NeuralNet(int input_nodes, int output_nodes, int hidden_nodes_per_layer, int hidden_layers=1, Hyperparameters hyper_params=DEFAULT_HYPERPARAMS)
        { 
            printf("Initializing the neural network with random weights and biases\n");
            // store the dimensions of the neural net and the hyperparameters
            this->n_input_nodes = input_nodes;
            this->n_output_nodes = output_nodes;
            this->n_hidden_nodes = hidden_nodes_per_layer;
            this->n_hidden_layers = hidden_layers;
            
            // print out some information about the type of neural net that was created
            describe();

            this->hyper_params = hyper_params;
            this->_output_layer_number = n_hidden_layers + 1;

            // ensure that hidden_layers is 1 
            if (n_hidden_layers != 1){
                printf("The number of hidden layers must be 1. The number of hidden layers is: %d", n_hidden_layers);
                exit(1);
            }

            // sigmoid as _activation func
            // this->_hidden_layer_activation_func = fast_sigmoid;
            // this->_hidden_layer_activation_func_deriv = sigmoid_derivative;
            // this->_output_activation_func = softmax;
            // this->_output_activation_func_deriv = softmax_derivative;

            this->_hidden_layer_activation_func = logistic_func;
            this->_hidden_layer_activation_func_deriv = logistic_func_derivative;
            this->_output_activation_func = logistic_func;
            this->_output_activation_func_deriv = logistic_func_derivative;

            // allocate memory and initialize the weights and biases with random values from 0 to 1
            _hidden_biases = random_matrix_0_to_1_factory(n_hidden_layers, n_hidden_nodes);
            _output_biases = random_vector_0_to_1_factory(n_output_nodes);

            _hidden_biases = zeros_matrix_factory(n_hidden_layers, n_hidden_nodes);
            _output_biases = zeros_vector_factory(n_output_nodes);
            
            // initialize the weights
            _input_weights = random_matrix_0_to_1_factory(n_hidden_nodes, n_input_nodes);
            if (n_hidden_layers > 1){
                Matrix* _hidden_weights = new Matrix[n_hidden_layers-1];
                for (int i = 0; i < n_hidden_layers-1; i++){
                    _hidden_weights[i] = random_matrix_0_to_1_factory(n_hidden_nodes, n_hidden_nodes);
                }
            }
            else{
                _hidden_weights = NULL;
            }
            _output_weights = random_matrix_0_to_1_factory(n_output_nodes, n_hidden_nodes);

            printf("Neural net initialized!\n");
            if (n_hidden_layers < 2 and n_hidden_nodes < 10){
                print_weights_and_biases();
            }
        }

        void describe(){
            // perceptron
            if (n_hidden_layers == 0){
                printf("Perceptron with %d input nodes and %d output nodes\n", n_input_nodes, n_output_nodes);
            }
            // single hidden layer
            else if (n_hidden_layers == 1){
                printf("Single hidden layer neural net with %d input nodes, %d hidden nodes, and %d output nodes\n", n_input_nodes, n_hidden_nodes, n_output_nodes);
            }
            // multiple hidden layers
            else{
                printf("Neural net with %d input nodes, %d hidden nodes, %d hidden layers, and %d output nodes\n", n_input_nodes, n_hidden_nodes, n_hidden_layers, n_output_nodes);
                // auto encoder
                if (n_input_nodes > n_hidden_nodes){
                    printf("There are less nodes in the hidden layer than the input layer. This is an autoencoder\n");
                }
            }
        }

        void print_weights_and_biases(){
            _hidden_biases.print("hidden biases");
            _output_biases.print("output biases");
            _input_weights.print("input weights");
            for (int i = 0; i < n_hidden_layers-1; i++){
                _hidden_weights[i].print("hidden weights");
            }
            _output_weights.print("output weights");

        }

        // loop over the data and aggregate the nn output as well as the layer outputs and return
        // note that this was easy to write with it copying each prediction into a new array. Could be optimized later.
        // likely not a computational bottleneck though
        Vector* feed_forward_bulk(double **features, int n_predictions, bool verbose=false)
        {
            Vector* predictions = new Vector[n_predictions];

            for (int i = 0; i < n_predictions; i++){
                Vector this_row(features[i], n_input_nodes);
                auto ff = feed_forwards(this_row);

                // store the data in the struct
                predictions[i] = ff.output;
                if (verbose){
                    this_row.print("Input");
                    ff.output.print("Prediction for row");
                }            
            }
            return predictions;
        }

        // ask the neural network to make a prediction given some input data
        FeedForwardData feed_forwards(Vector X){

            // check dimensions
            if (X.length != n_input_nodes){
                printf("Error: input data has %d rows, but the neural net has %d input nodes\n", X.length, n_input_nodes);
                exit(1);
            }

            Vector* layer_outputs = new Vector[n_hidden_layers];

            // variable a for the output of the current layer
            Vector a = _get_layer_output(X, 1);       // this matrix should only have 1 row (essentially a vector)
            layer_outputs[0] = a;
            // a.print("a");
            // calculate through all the hidden layers looping over each hidden layer
            for (int i = 0; i < n_hidden_layers-1; i++){
                a = _get_layer_output(a, i+2);
                layer_outputs[i+1] = a;
            }

            // calculate the output
            a = _get_layer_output(a, n_hidden_layers+1);
            // a.print("a");
            return FeedForwardData{layer_outputs, a};
        }

        // makes predictions for the full dataset, does backpropogation on that batch mutating the weights and biases, iterates, and returns the losses 
        Vector train(LabeledData data, int max_epochs=1000){
            // validate that data shape matches inputs and outputs
            if (data.n_features != n_input_nodes){
                printf("Error: data has %d features but the neural network has %d input nodes\n", data.n_features, n_input_nodes);
                exit(1);
            }
            if (data.n_outputs != n_output_nodes){
                printf("Error: data has %d labels but the neural network has %d output nodes\n", data.n_outputs, n_output_nodes);
                exit(1);
            }

            printf("Training the neural network for %d epochs\n", max_epochs);
            // variable losses_arr set initially to array of all zeros
            double* losses_arr = new double[max_epochs];

            // var to hold the loss over the course of training
            // calculate the output of the neural network
            for (int i=0; i < max_epochs; i++){
                losses_arr[i] = _backpropogate_full_data_set(data, i);
            }
            Vector losses(losses_arr, max_epochs);
            return losses;
        }

        // save weights and biases to a file. 
        void save_to_file(){
            // a meta file will be needed to store the dimensions of the network and hyperparams
            ofstream myfile ("model/meta_info.txt");
                if (myfile.is_open())
                {
                    // save the dimensions of the network
                    myfile << n_input_nodes << "\n";
                    myfile << n_hidden_layers << "\n";
                    myfile << n_hidden_nodes << "\n";
                    myfile << n_output_nodes << "\n";
                    myfile << hyper_params.learning_rate << "\n";
                    myfile << hyper_params.momentum << "\n";
                    
                    myfile.close();
                }
                else cout << "Unable to open file";
            printf("Saved to file %s\n", "meta_info.txt");

            _input_weights.save_to_file("model/input_weights.csv");
            for (int i = 0; i < n_hidden_layers-1; i++){
                _hidden_weights[i].save_to_file("model/hidden_weights.csv");
            }
            _output_weights.save_to_file("model/output_weights.csv");
            if (n_hidden_layers > 0){
                _hidden_biases.save_to_file("model/hidden_biases.csv");
            }
            _output_biases.save_to_file("model/output_biases.csv");
        }

        // load weights and biases from a file taking the folder as an optional argument
        void load_from_file(std::string folder="model"){
            // get the dimensions from the metadata file
            std::ifstream metadata_file(folder + "/meta_info.txt");
            std::string line;
            std::getline(metadata_file, line);
            n_input_nodes = std::stoi(line);
            std::getline(metadata_file, line);

            n_hidden_layers = std::stoi(line);
            std::getline(metadata_file, line);
            n_hidden_nodes = std::stoi(line);

            std::getline(metadata_file, line);
            n_output_nodes = std::stoi(line);

            // get the weights and biases from the weights and biases files
            _input_weights = get_matrix_from_csv("model/input_weights.csv");
            _hidden_weights = new Matrix[n_hidden_layers-1];
            _output_weights = get_matrix_from_csv("model/output_weights.csv");

            _hidden_biases = get_matrix_from_csv("model/hidden_biases.csv");
            _output_biases = get_matrix_from_csv("model/output_biases.csv").get_row(0);

            // ensure the   dimensions are correct
            if (_input_weights.cols != n_input_nodes){
                printf("Error: input weights have %d columns but the neural network has %d input nodes\n", _input_weights.cols, n_input_nodes);
                exit(1);
            }
            if (_input_weights.rows != n_hidden_nodes){
                printf("Error: input weights have %d rows but the neural network has %d hidden nodes\n", _input_weights.rows, n_hidden_nodes);
                exit(1);
            }
            if (_output_weights.cols != n_hidden_nodes){
                printf("Error: output weights have %d columns but the neural network has %d hidden nodes\n", _output_weights.cols, n_hidden_nodes);
                exit(1);
            }
            if (_output_weights.rows != n_output_nodes){
                printf("Error: output weights have %d rows but the neural network has %d output nodes\n", _output_weights.rows, n_output_nodes);
                exit(1);
            }

            printf("Loaded all weights and biases from file");
            print_weights_and_biases();
        }

        // destructor
        ~NeuralNet(){
            printf("Destroying the neural network");
            // free the memory for the weights and biases
            // delete[] _hidden_weights;
        }

    private:
        // commonly known as "z" the output of the layer before application of the activation function
        // 2D arrays are used here because the matrix_multiply function works with that size. 
        // layer_input is a 2D array with dimensions (1, n_nodes_in_layer)
        Vector _do_layer_calcs(Vector input, Matrix weights, Vector biases, double (*activation_func)(double)){
            Vector z = add_vectors(matrix_vector_multiply(weights, input), biases);
            // z.print("z");
            z.apply_function(activation_func);
            return z;
        }

        Vector _get_layer_output(Vector input_to_layer, int n_layer){
            // printf("\ngetting layer output for layer %d\n", n_layer);
            if (n_layer == 1){                          // input layer
                return _do_layer_calcs(input_to_layer, _input_weights, _hidden_biases.get_row(0), _hidden_layer_activation_func);
            }
            else if (n_layer == _output_layer_number){  // output layer
                return _do_layer_calcs(input_to_layer, _output_weights, _output_biases, _output_activation_func);
            }
            else{                                       // hidden layers
                return _do_layer_calcs(input_to_layer, _hidden_weights[n_layer-2], _hidden_biases.get_row(n_layer-1), _hidden_layer_activation_func);
            }
        }

        // loops over the data set, creates minibatches and does backpropogation on each batch. returns mean loss on the batch
        double _backpropogate_full_data_set(LabeledData data, int epoch){
            // calculate the number of batches
            int n_batches = (int) ceil((double)data.n_samples / (double) hyper_params.minibatch_size);
            // int n_batches = 1;
            // printf("Number of batches: %d\n", n_batches);
            double loss = 0;

            // loop over the batches
            for (int i = 0; i < n_batches; i++){
                // get the start and end indices for the batch
                int start = i * hyper_params.minibatch_size;
                int end = min(start + hyper_params.minibatch_size, data.n_samples);

                // LabeledData batch = get_data_subset(data, start, end);
                
                // update the loss and the weights and biases
                BackPropData gradients = _backpropogate_mini_batch(data);
                loss += gradients.loss;

                // normalize the dervatives of weights and biases for the batch
                BackPropData delta = get_scaled_copy_of_backpropdata(gradients, -hyper_params.learning_rate/data.n_samples);

                BackPropData actual_weights = {0, _input_weights, _hidden_weights, _output_weights, _hidden_biases, _output_biases};
                back_prop_weights_and_biases_plus_equals(actual_weights, delta);
                if (hyper_params.momentum > 0){
                    if (i > 0 || epoch > 0){
                        BackPropData momentum_gradients = get_scaled_copy_of_backpropdata(_prior_gradients, hyper_params.momentum);
                        back_prop_weights_and_biases_plus_equals(actual_weights, momentum_gradients);
                    }
                }
                _prior_gradients = gradients;
            }
            loss /= n_batches;  // this is now the average for the epoch
            return loss;
        }

        // calculate and return the loss and the gradients for a mini batch of data
        BackPropData _backpropogate_mini_batch(LabeledData data){

            BackPropData gradients_for_batch;   // this is an average of the gradients for each sample in the batch
            // loop over the data doing backpropogation for each single sample
            for (int i = 0; i < data.n_samples; i++){
                Vector this_input(data.features[i], n_input_nodes);
                Vector label(data.labels[i], n_output_nodes);
                auto gradients = _backpropogate_single_sample(this_input, label);
                
                // print the gradients
                // gradients.dBOutput.print("dBOutput");
                // gradients.dWOutput.print("dWOutput");
                // gradients.dBHidden.print("dBHidden");
                // gradients.dWInput.print("dWInput");
                
                // add the values to the rest of the matrices
                if (i == 0){
                    gradients_for_batch = gradients;
                }
                else{
                    gradients_for_batch.loss += gradients.loss;
                    back_prop_weights_and_biases_plus_equals(gradients_for_batch, gradients);
                }
            }

            // normalize the loss by the number of samples
            gradients_for_batch.loss = gradients_for_batch.loss / data.n_samples;

            return gradients_for_batch;
        }

        // gradient descent for a single sample assuming there is only the output weights and the output biases
        BackPropData _backpropogate_single_sample(Vector X, Vector y){
            
            // calculate the output of the neural network and the loss
            auto prediction = feed_forwards(X);

            // dW_L = dzOutput * a(L-1)^T; where a = g(z) = g(Wx + b) and L is the number of layers
            // NOTATION is dx = d(loss)/d(x)
            // calculate the derivative of the loss with respect to the output
            Vector error = subtract_vectors(prediction.output, y);  // in some cases you would multiply this by 2 for the derivative of sum of square errors, but not necessary
            Vector dWoutput_term2 = prediction.output.copy();
            dWoutput_term2.apply_function(_output_activation_func_deriv);
            Vector dZOutput = hadamard_product(error, dWoutput_term2);  // elementwise multiplication
            Matrix dWOutput = multiply_vectors_for_matrix(dZOutput, prediction.layer_outputs[n_hidden_layers-1]);

            // now do the hidden layers
            Matrix dWHidden[n_hidden_layers-1];
            for (int i = n_hidden_layers-1; i > 0; i--){
            }


            // now do the input layer
            Vector dWinput_term1 = prediction.layer_outputs[0].copy();
            dWinput_term1.apply_function(_hidden_layer_activation_func_deriv);
            Vector dWinput_term2 = matrix_vector_multiply(_output_weights.transpose(), dZOutput);   // note that the order here is switched around in a weird way...
            Vector* dZHidden = new Vector[n_hidden_layers];
            dZHidden[0] = hadamard_product(dWinput_term1, dWinput_term2);
            Matrix dWInput = multiply_vectors_for_matrix(dZHidden[0], X);    // swapped order here to make matrix dims work...

            return BackPropData{
                sum_of_squared_errors(prediction.output, y),
                dWInput, 
                dWHidden,
                dWOutput, 
                aggregate_vectors(dZHidden, n_hidden_layers),
                dZOutput,
                n_hidden_layers};
        }

        // the weights and the biases of the neural net sized according to the number of nodes in each layers these are all dynamically allocated arrays
        Matrix _hidden_biases;
        Vector _output_biases;
        
        Matrix _input_weights;
        Matrix* _hidden_weights;
        Matrix _output_weights;

        BackPropData _prior_gradients;  // used for momentum

        int _output_layer_number;

        // an array of pointers to functions taking and returning doubles (for each hidden layer)
        double (*_hidden_layer_activation_func)(double);
        // array for the derivative of the activation functions
        double (*_hidden_layer_activation_func_deriv)(double);

        // the activation function to be used for the output layer (expecting functions that take and return doubles)
        double (*_output_activation_func)(double);
        double (*_output_activation_func_deriv)(double);

};
