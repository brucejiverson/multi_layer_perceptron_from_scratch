#include<iostream>
#include<random>

#include"linear_algebra.cpp"
#include"machine_learning_helpers.cpp"
using namespace std;


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
    Matrix* dWHidden_copy = new Matrix[data.n_hidden_layers];
    for (int i=0; i<data.n_hidden_layers-1; i++){
        dWHidden_copy[i] = data.dWHidden[i].copy();
    }
    BackPropData copy = {
        data.loss, data.dWInput.copy(), 
        dWHidden_copy,
        data.dWOutput.copy(), 
        data.dBHidden.copy(), 
        data.dBOutput.copy()};
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
    data1.dBOutput = (data1.dBOutput, data2.dBOutput);
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
        int n_output_nodes;
        int n_hidden_layers = 1;
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
                    // _hidden_weights[i] = new double*[n_hidden_nodes];
                    _hidden_weights[i] = random_matrix_0_to_1_factory(n_hidden_nodes, n_hidden_nodes);
                }
            }
            else{
                _hidden_weights = NULL;
            }
            _output_weights = random_matrix_0_to_1_factory(n_output_nodes, n_hidden_nodes);

            printf("Neural net initialized:\n");
            print_weights_and_biases();
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
        Vector* feed_forward_bulk(double **features)
        {
            // loop over all of the features and make a prediction for each one
            int n_predictions = sizeof(features) / sizeof(features[0]);

            Vector* predictions = new Vector[n_predictions];

            for (int i = 0; i < n_predictions; i++){
                Vector this_row(features[i], n_input_nodes);
                auto ff = feed_forwards(this_row);

                // store the data in the struct
                predictions[i] = ff.output;
            }
            return predictions;
        }

        // ask the neural network to make a prediction given some input data
        FeedForwardData feed_forwards(Vector X){
            // note that the arrays for storing the outputs have an extra dimension. 
            //This is to aid bulk predictions in higher level functions keeping same return type

            Vector* layer_outputs = new Vector[n_hidden_layers+1];

            // variable a for the output of the current layer
            Vector a = _get_layer_output(X, 1);       // this matrix should only have 1 row (essentially a vector)
            layer_outputs[0] = a;
            
            // calculate through all the hidden layers looping over each hidden layer
            for (int i = 0; i < n_hidden_layers-1; i++){
                a = _get_layer_output(a, i+2);
                layer_outputs[i+1] = a;
            }

            // calculate the output
            a = _get_layer_output(a, n_hidden_layers+1);
            layer_outputs[n_hidden_layers] = a;

            return FeedForwardData{layer_outputs, a};
        }

        // makes predictions for the full dataset, does backpropogation on that batch mutating the weights and biases, iterates, and returns the losses 
        Vector train(LabeledData data, int max_epochs=1000){
            
            printf("Training the neural network for %d epochs\n", max_epochs);
            // variable losses_arr set initially to array of all zeros
            double* losses_arr = new double[max_epochs];

            Vector losses(losses_arr, max_epochs);
            // var to hold the loss over the course of training
            // calculate the output of the neural network
            for (int i=0; i < max_epochs; i++){
                losses.array[i] = _backpropogate_full_data_set(data);
            }
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

        // destructor
        ~NeuralNet()
        {
            printf("Destroying the neural network");
            // free the memory for the weights and biases
            delete[] _hidden_weights;
        }

    private:
        // commonly known as "z" the output of the layer before application of the activation function
        // 2D arrays are used here because the matrix_multiply function works with that size. 
        // layer_input is a 2D array with dimensions (1, n_nodes_in_layer)
        Vector _do_layer_calcs(Vector input, Matrix weights, Vector biases, double (*activation_func)(double)){
            Vector z = add_vectors(matrix_vector_multiply(weights, input), biases);
            z.print("z");
            z.apply_function(activation_func);
            return z;
        }

        Vector _get_layer_output(Vector input_to_layer, int n_layer)
        {
            printf("\ngetting layer output for layer %d\n", n_layer);
            // switch case for layer number. Each case has following steps: allocate memory, multiply inputs and weights, add biases, and apply activation function
            if (n_layer == 1){
                return _do_layer_calcs(input_to_layer, _input_weights, _hidden_biases.get_row(0), _hidden_layer_activation_func);
            }
            else if (n_layer == _output_layer_number){  // output layer
                return _do_layer_calcs(input_to_layer, _output_weights, _output_biases, _output_activation_func);
            }
            else{   // hidden layers
                return _do_layer_calcs(input_to_layer, _hidden_weights[n_layer-2], _hidden_biases.get_row(n_layer-1), _hidden_layer_activation_func);
            }
        }

        // loops over the data set, creates minibatches and does backpropogation on each batch. returns mean loss on the batch
        double _backpropogate_full_data_set(LabeledData data){
            // calculate the number of batches
            int n_batches = (int) ceil((double)data.n_samples / (double) hyper_params.minibatch_size);
            printf("Number of batches: %d", n_batches);
            
            double loss = 0;
            BackPropData prior_gradients;

            // loop over the batches
            for (int i = 0; i < n_batches; i++){
                // get the start and end indices for the batch
                int start = i * hyper_params.minibatch_size;
                int end = min(start + hyper_params.minibatch_size, data.n_samples);

                LabeledData batch = get_data_subset(data, start, end);
                // do calcs for the batch
                BackPropData gradients = _backpropogate_mini_batch(batch);

                // set the gradients to be zero for the biases (REMOVE LATER)
                gradients.dBOutput = _output_biases;
                gradients.dBHidden = _hidden_biases;
                
                // update the loss and the weights and biases
                loss += gradients.loss;
                BackPropData delta = get_scaled_copy_of_backpropdata(gradients, -1.0*hyper_params.learning_rate);
                if(i > 0){ // momentum term
                    BackPropData momentum = get_scaled_copy_of_backpropdata(prior_gradients, hyper_params.momentum);
                    back_prop_weights_and_biases_plus_equals(delta, momentum);  // add momentum to delta
                }
                matrix_plus_equal(_input_weights, delta.dWInput);
                matrix_plus_equal(_output_weights, delta.dWOutput);
                for (int j = 0; j < n_hidden_layers-1; j++){
                    matrix_plus_equal(_hidden_weights[j], delta.dWHidden[j]);
                }
                vector_plus_equal(_output_biases, delta.dBOutput);
                matrix_plus_equal(_hidden_biases, delta.dBHidden);

                prior_gradients = gradients;
            }
            loss /= n_batches;  // this is now the average for the epoch
            return loss;
        }

        // calculate and return the loss and the gradients for a mini batch of data
        BackPropData _backpropogate_mini_batch(LabeledData data){

            BackPropData back_prop_data_batch = {
                0, 
                Matrix(), 
                new Matrix[n_hidden_layers], 
                Matrix(), 
                Matrix(), 
                Vector(), 
                n_hidden_layers};

            // loop over the data doing backpropogation for each single sample
            for (int i = 0; i < data.n_samples; i++){
                Vector this_input(data.features[i], n_input_nodes);
                Vector label(data.labels[i], n_output_nodes);
                auto back_prop_data = _backpropogate_single_sample(this_input, label);
                back_prop_data_batch.loss += back_prop_data.loss;
                // add the values to the rest of the matrices
                if (i == 0){
                    back_prop_data_batch = back_prop_data;
                }
                else{
                    back_prop_weights_and_biases_plus_equals(back_prop_data_batch, back_prop_data);
                }
            }

            // normalize the loss by the number of samples
            back_prop_data_batch.loss = back_prop_data_batch.loss / data.n_samples;

            // normalize the dervatives of weights and biases for the batch 
            BackPropData delta = get_scaled_copy_of_backpropdata(back_prop_data_batch, 1.0/data.n_samples);

            return delta;
        }

        // gradient descent for a single sample assuming there is only the output weights and the output biases
        BackPropData _backpropogate_single_sample(Vector X, Vector y){
            
            // calculate the output of the neural network and the loss
            auto prediction = feed_forwards(X);

            // NOTATION is dx = d(loss)/d(x)

            // calculate the derivative of the loss with respect to the output
            Vector error = subtract_vectors(prediction.output, y);     // 1D
            error.scale(2);
            Vector dWoutput_term2 = prediction.output.copy();
            dWoutput_term2.apply_function(_output_activation_func_deriv);    // 1D
            Vector dZOutput = hadamard_product(error, dWoutput_term2);

            // dW_L = dzOutput * a(L-1)^T; where a = g(z) = g(Wx + b) and L is the number of layers
            Matrix dWOutput = multiply_vectors_for_matrix(dZOutput, prediction.layer_outputs[n_hidden_layers-1]);

            // now do the input layer
            Vector dWinput_term1 = prediction.layer_outputs[0].copy();
            dWinput_term1.apply_function(_hidden_layer_activation_func_deriv);    // 1D
            Vector dWinput_term2 = matrix_vector_multiply(_output_weights.transpose(), dZOutput);   // note that the order here is switched around in a weird way...
            Vector* dZHidden = new Vector[n_hidden_layers];
            dZHidden[0] = hadamard_product(dWinput_term1, dWinput_term2);
            Matrix dWInput = multiply_vectors_for_matrix(dZHidden[0], X);    // swapped order here to make matrix dims work...
        
            Matrix dWHidden[n_hidden_layers-1];

            return BackPropData{
                sum_of_squared_errors(prediction.output.array, y.array), 
                dWInput, 
                dWHidden, 
                dWOutput, 
                aggregate_vectors(dZHidden, n_hidden_layers),
                dZOutput};
        }

        // the weights and the biases of the neural net sized according to the number of nodes in each layers these are all dynamically allocated arrays
        Matrix _hidden_biases;
        Vector _output_biases;
        
        Matrix _input_weights;
        Matrix* _hidden_weights;
        Matrix _output_weights;

        int _output_layer_number;

        // an array of pointers to functions taking and returning doubles (for each hidden layer)
        double (*_hidden_layer_activation_func)(double);
        // array for the derivative of the activation functions
        double (*_hidden_layer_activation_func_deriv)(double);

        // the activation function to be used for the output layer (expecting functions that take and return doubles)
        double (*_output_activation_func)(double);
        double (*_output_activation_func_deriv)(double);

};