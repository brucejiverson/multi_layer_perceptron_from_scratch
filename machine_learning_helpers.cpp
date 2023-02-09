#include<iostream>
#include <math.h>       /* pow */


struct LabeledData
{
    double** features;
    double** labels;
    int n_features;
    int n_outputs;
    int n_samples;
};


LabeledData get_data_subset(LabeledData data, int start_index, int end_index){
    int n_samples = end_index - start_index;
    double** features = new double*[n_samples];
    double** labels = new double*[n_samples];
    for (int i = 0; i < n_samples; i++){
        features[i] = data.features[start_index + i];
        labels[i] = data.labels[start_index + i];
    }
    return LabeledData{features, labels, data.n_features, data.n_outputs, n_samples};
}


double relu(double x){
    if (x < 0){
        return 0;
    }
    else{
        return x;
    }
}


double relu_derivative(double x){
    if (x < 0){
        return 0;
    }
    else{
        return 1;
    }
}


// the most common sigmoid function
double logistic_func(double x){
    return 1 / (1 + exp(-x));
}


double logistic_func_derivative(double x){
    // return logistic_func(x) * (1 - logistic_func(x));
    return x * (1 - x);
}


double fast_sigmoid(double x){
    return x / (1 + abs(x));
}


double sigmoid_derivative(double x){
    return fast_sigmoid(x) * (1 - fast_sigmoid(x));
}


double tanh(double x){
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}


int unit_step_function(double x){
    if (x < 0){
        return 0;
    }
    else{
        return 1;
    }
}


double tanh_derivative(double x){
    return 1 - pow(tanh(x), 2);
}


double softmax(double x){
    return exp(x) / (exp(x) + exp(-x));
}


double softmax_derivative(double x){
    return softmax(x) * (1 - softmax(x));
}


double sum_of_squared_errors(Vector predictions, Vector labels){
    if (predictions.length != labels.length){
        printf("The predictions and labels are not the same size. Predictions size: %d, Labels size: %d", predictions.length, labels.length);
        exit(1);
    }
    // ensure that the predictions and labels are the same size
    double sum = 0;
    for (int i = 0; i < predictions.length; i++){
        sum += pow(predictions.array[i] - labels.array[i], 2)/2;
    }
    return sum;
}


double F1_score(double *predictions, double *labels)
    {
        printf("Scoring the predictions");
        // F1 score which is: 2 * (precision * recall) / (precision + recall)
        // precision is: true positives / (true positives + false positives)
        // recall is: true positives / (true positives + false negatives)

        // ensure that the predictions and labels are the same size
        int vector_size = (int)(sizeof(predictions) / sizeof(predictions[0]));
        int label_size = (int)(sizeof(labels) / sizeof(labels[0]));
        if (vector_size != label_size)
        {
            printf("The predictions and labels are not the same size. Predictions size: %d, Labels size: %d", vector_size, label_size);
        }

        // initialize the true positives, false positives, true negatives, and false negatives
        int true_positives = 0;
        int false_positives = 0;
        int true_negatives = 0;
        int false_negatives = 0;

        // loop through the predictions and labels and count the true positives, false positives, true negatives, and false negatives
        for (int i = 0; i < vector_size; i++){
            // if the prediction is 1 and the label is 1, then it is a true positive
            if (predictions[i] == 1 && labels[i] == 1){
                true_positives++;
            }
            // if the prediction is 1 and the label is 0, then it is a false positive
            else if (predictions[i] == 1 && labels[i] == 0){
                false_positives++;
            }
            // if the prediction is 0 and the label is 0, then it is a true negative
            else if (predictions[i] == 0 && labels[i] == 0){
                true_negatives++;
            }
            // if the prediction is 0 and the label is 1, then it is a false negative
            else if (predictions[i] == 0 && labels[i] == 1){
                false_negatives++;
            }
        }

        // calculate the precision and recall
        double precision = (double)true_positives / (true_positives + false_positives);
        double recall = (double)true_positives / (true_positives + false_negatives);

        // calculate the F1 score
        double f1_score = 2 * (precision * recall) / (precision + recall);

        printf("F1 score: %f", f1_score);
        return f1_score;
    }


double F2_score(double *predictions, double *labels)
    {
        printf("Scoring the predictions");
        // F2 score which is: (1 + 2^2) * (precision * recall) / (2^2 * precision + recall)
        // precision is: true positives / (true positives + false positives)
        // recall is: true positives / (true positives + false negatives)

        // ensure that the predictions and labels are the same size
        int vector_size = (int)(sizeof(predictions) / sizeof(predictions[0]));
        int label_size = (int)(sizeof(labels) / sizeof(labels[0]));
        if (vector_size != label_size)
        {
            printf("The predictions and labels are not the same size. Predictions size: %d, Labels size: %d", vector_size, label_size);
        }

        // initialize the true positives, false positives, true negatives, and false negatives
        int true_positives = 0;
        int false_positives = 0;
        int true_negatives = 0;
        int false_negatives = 0;

        // loop through the predictions and labels and count the true positives, false positives, true negatives, and false negatives
        for (int i = 0; i < vector_size; i++){
            // if the prediction is 1 and the label is 1, then it is a true positive
            if (predictions[i] == 1 && labels[i] == 1){
                true_positives++;
            }
            // if the prediction is 1 and the label is 0, then it is a false positive
            else if (predictions[i] == 1 && labels[i] == 0){
                false_positives++;
            }
            // if the prediction is 0 and the label is 0, then it is a true negative
            else if (predictions[i] == 0 && labels[i] == 0){
                true_negatives++;
            }
            // if the prediction is 0 and the label is 1, then it is a false negative
            else if (predictions[i] == 0 && labels[i] == 1){
                false_negatives++;
            }
        }

        // calculate the precision and recall
        double precision = (double)true_positives / (true_positives + false_positives);
        double recall = (double)true_positives / (true_positives + false_negatives);

        // calculate the F2 score
        double f2_score = (1 + pow(2, 2)) * (precision * recall) / (pow(2, 2) * precision + recall);
        
        printf("F2 score: %f", f2_score);
        return f2_score;
    }
