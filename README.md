# Idea and Criteria for Neural Net from scratch
I am looking to refresh some of my C++ knowledge, and have a long standing interest in machine learning. Building a neural net from scratch seems like an excellent project that is sized appropriately and will deepen my understanding of machine learning. 

I don't expect this code to outperform anyone, nor do I intend to use this code for anything past demonstrations of functionality. 

Most blog posts with basic neural net projects seem to be for single layer networks and are all written in python. I consider that to be trivial, so this project will have **fully connected networks of arbitrary input, output, hidden nodes, and hidden layer numbers**. This and the fact that I've written less than 50 lines of C++ last year should make this project sufficiently challenging to hold my interest. 

# Requirements
1. Basic neural net with 1 hidden layer and gradient descent backpropogation. All layers dense/fully connected
2. Must do all math myself: no libraries, including std::vector, arrays only
3. Configurable number of input and output nodes
4. Configurable activation functions as activation function selection can be problem specific
5. Bonus points: 
	1. Configurable number of hidden layers
	2. Implement some common hyperaparameters such as learning rate, momentum, batch size
	3. Test on larger/more complicated data sets
	4. Can save/load models from disk

I don't want to hassel with figuring out the C++ bindings for matplotlib or other C++ plotting techniques, so instead I've exported the data and used a python script for data visualization. 

# Demonstration
![layer output](./images/1_layer_output_XOR.png)
![loss](./images/2_hidden_layers_10_hidden_nodes_XOR_.5_0.png)


# How to Run
Set up a quick python envrionment, install matplotlib, and execute the following in terminal:
`python run.py -i xor_test`

This python script will compile the xor_test c++ file, run the executable, and plot the losses from training. 
See my website https://www.bruceiverson.com/portfolio-collections/my-portfolio/mutli-layer-perceptron for more details.
