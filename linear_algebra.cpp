#include <iostream>
#include <random>
#include <fstream>
using namespace std;


void print_1D_array(double* array, int length){
    for(int i=0; i<length; i++){
        printf("%.4f ", array[i]);
    }
    std::cout<<std::endl;
}


void print_2D_array(double** array, int rows, int cols){
    // print the result looping to the rows and columns
    for(int i=0; i<rows; i++){
        print_1D_array(array[i], cols);
    }
    std::cout<<std::endl;
}


class Vector{
    public:
        double* array;
        int length;

        // default constructor
        Vector(){
            this->array = 0;
            this->length = 0;
        }

        // constructor
        Vector(double* array, int l){
            this->array = array;
            this->length = l;
        }

        // get element
        double get_element(int index){
            return array[index];
        }

        // apply function to transform each element in the vector
        void apply_function(double (*func)(double)){
            for (int i = 0; i < length; i++){
                array[i] = func(array[i]);
            }
        }

        void scale(double scaler){
            for (int i = 0; i < length; i++){
                array[i] *= scaler;
            }
        }

        Vector copy(){
            double* arr = new double[length];
            for (int i = 0; i < length; i++){
                arr[i] = array[i];
            }
            return Vector(arr, length);
        }
        
        // a function called print that takes in the name of the array as an argument and prints it
        void print(std::string name=""){
            // if (describe){
                printf("Vector %s w/ length %d is:\n", name.c_str(), length);
            // }
            print_1D_array(array, length);
            printf("\n");
        }

        // takes the files name as a string
        void save_to_file(std::string filename){
            ofstream myfile (filename);
                if (myfile.is_open())
                {
                    for(int count = 0; count < length; count ++){
                        myfile << array[count] << "," ;
                    }
                    myfile.close();
                }
                else cout << "Unable to open file";
            printf("Saved to file %s\n", filename.c_str());
        }

};


// a matrix class that holds an array, the dimensions, and has a transpose method
class Matrix{
    public:
        double** array;
        int rows;
        int cols;

        // default constructor
        Matrix(){
            this->array = 0;
            this->rows = 0;
            this->cols = 0;
        }

        // constructor
        Matrix(double** array, int r, int c){
            this->array = array;
            this->rows = r;
            this->cols = c;
        }

        // get element
        double get_element(int row, int col){
            return array[row][col];
        }

        // get row
        double* get_row_as_arr(int row){
            return array[row];
        }

        // get column
        double* get_col_as_arr(int col){
            double* arr = new double[rows];
            for (int i = 0; i < rows; i++){
                arr[i] = array[i][col];
            }
            return arr;
        }

        Vector get_row(int row){
            return Vector(get_row_as_arr(row), cols);
        }

        Vector get_column(int col){
            return Vector(get_col_as_arr(col), rows);
        }

        // multiplies the matrix in place by a scalar
        void scale(double scalar){
            for (int i = 0; i < rows; i++){
                for (int j = 0; j < cols; j++){
                    array[i][j] = array[i][j] * scalar;
                }
            }
        }

        // returns a new matrix that is the transpose of the current matrix
        Matrix transpose(){
            // create a new array to store the transposed matrix
            double** arr = new double*[cols];
            for (int i = 0; i < cols; i++){
                arr[i] = new double[rows];
            }

            // transpose the matrix
            for (int i = 0; i < rows; i++){
                for (int j = 0; j < cols; j++){
                    arr[j][i] = array[i][j];
                }
            }
            return Matrix(arr, cols, rows);
        }

        // apply function to transform each element in the matrix
        void apply_function(double (*func)(double)){
            for (int i = 0; i < rows; i++){
                for (int j = 0; j < cols; j++){
                    array[i][j] = func(array[i][j]);
                }
            }
        }

        // a function called print that takes in the name of the array as an argument and prints it
        void print(std::string name=""){;
            printf("Matrix %s sized %d x %d is:\n", name.c_str(), rows, cols);
            print_2D_array(array, rows, cols);
        }

        Matrix copy(){
            double** arr = new double*[rows];
            for (int i = 0; i < rows; i++){
                arr[i] = new double[cols];
                for (int j = 0; j < cols; j++){
                    arr[i][j] = array[i][j];
                }
            }
            return Matrix(arr, rows, cols);
        }

        void save_to_file(std::string filename){
            ofstream myfile (filename);
                if (myfile.is_open())
                {
                    for(int count = 0; count < rows; count ++){
                        for(int count2 = 0; count2 < cols; count2 ++){
                            myfile << array[count][count2] << "," ;
                        }
                        myfile << ",\n";
                    }
                    myfile.close();
                }
                else cout << "Unable to open file";
            printf("Saved to file %s\n", filename.c_str());
        }
};


Matrix get_matrix_from_csv(std::string filename){
    // open the file
    ifstream file(filename);
    // create a string to hold each line
    string line;
    // create a vector to hold the lines
    vector<string> lines;
    // read the file line by line
    while (getline(file, line)){
        lines.push_back(line);
    }
    // get the number of rows and columns
    int rows = lines.size();
    int cols = 0;
    for (int i = 0; i < lines[0].size(); i++){
        if (lines[0][i] == ','){            
            cols++;
        }
    }

    if (lines[0][lines[0].size()-1] == ',' && lines[0][lines[0].size()-2] == ','){
        cols--;
    }

    // create a 2D array to hold the matrix
    double** matrix = new double*[rows];
    for (int i = 0; i < rows; i++){
        matrix[i] = new double[cols];
    }
    // fill the matrix
    for (int i = 0; i < rows; i++){
        int col = 0;
        string num = "";
        for (int j = 0; j < lines[i].size(); j++){
            if (lines[i][j] == ',' && num != ""){
                matrix[i][col] = stod(num);
                num = "";
                col++;
            }
            else{
                num += lines[i][j]; // add the character to the string
            }
        }
    }
    return Matrix(matrix, rows, cols);
}


Matrix get_matrix_from_1D_array(double* arr, int length){
    double** matrix = new double*[1];
    matrix[0] = new double[length];
    for (int i = 0; i < length; i++){
        matrix[0][i] = arr[i];
    }
    return Matrix(matrix, 1, length);
}


// array size macros
#define ROWS_IN_2D_ARRAY(a2D)       (int)( sizeof( a2D       ) / sizeof( a2D[0]          )) // No. of Rows in a 2D array
#define COLUMNS_IN_2D_ARRAY(a2D)    (int)( sizeof( a2D[0]    ) / sizeof( a2D[0][0]       )) // No. of Columns in a 2D array
// #define GET_2DARRAY_DIMS(a2D)        ArrayDims{ROWS_IN_2D_ARRAY(a2D), COLUMNS_IN_2D_ARRAY(a2D)}
#define GET_ELEMENTS_IN_VECTOR(a1D) (int)( sizeof( a1D       ) / sizeof( a1D[0]          )) // No. of Elements in a 1D array


// matrix multiply function with Matrix class
Matrix matrix_multiply(Matrix matrix1, Matrix matrix2)
{
    // check that the number of columns in the first matrix is equal to the number of rows in the second matrix
    if (matrix1.cols != matrix2.rows)    {
        // print out the and an error message
        printf("The number of columns in the first matrix is not equal to the number of rows in the second matrix.\n");
        printf("Matrix 1: %d x %d. ", matrix1.rows, matrix1.cols);
        printf("Matrix 2: %d x %d.\n", matrix2.rows, matrix2.cols);
        return Matrix(0, 0, 0);
        }

    // create a new matrix to store the result of the multiplication
    double** arr = new double*[matrix1.rows];

    for (int i = 0; i < matrix1.rows; i++){        // iterate through the rows of the first matrix
        arr[i] = new double[matrix2.cols];
        for (int j = 0; j < matrix2.cols; j++){    // iterate through the columns of the second matrix
            arr[i][j] = 0;
            for (int k = 0; k < matrix1.cols; k++){// iterate through the columns of the first matrix
                arr[i][j] += matrix1.array[i][k] * matrix2.array[k][j];
            }
        }
    }
    return Matrix(arr, matrix1.rows, matrix2.cols); 
}

// subtracts the second matrix from the first matrix in poace
void matrix_minus_equals(Matrix matrix1, Matrix matrix2)
{
    // check that the dimensions of the matrices are the same
    if (matrix1.rows != matrix2.rows || matrix1.cols != matrix2.cols){
        // print out the and an error message
        printf("The dimensions of the matrices are not the same.\n");
        printf("Matrix 1: %d x %d. ", matrix1.rows, matrix1.cols);
        printf("Matrix 2: %d x %d.\n", matrix2.rows, matrix2.cols);
        return;
    }

    // subtract the second matrix from the first matrix
    for (int i = 0; i < matrix1.rows; i++){        // iterate through the rows of the first matrix
        for (int j = 0; j < matrix1.cols; j++){    // iterate through the columns of the second matrix
            matrix1.array[i][j] -= matrix2.array[i][j];
        }
    }
}

// modifies the first matrix to be the sum of the first and second matrices
void matrix_plus_equal(Matrix matrix1, Matrix matrix2)
{
    // check that the dimensions of the matrices are the same
    if (matrix1.rows != matrix2.rows || matrix1.cols != matrix2.cols){
        // print out the and an error message
        printf("Matrix 1: %d x %d. ", matrix1.rows, matrix1.cols);
        printf("Matrix 2: %d x %d.\n", matrix2.rows, matrix2.cols);
        throw runtime_error("The dimensions of the matrices are not the same.");
    }

    // add the second matrix to the first matrix
    for (int i = 0; i < matrix1.rows; i++){        // iterate through the rows of the first matrix
        for (int j = 0; j < matrix1.cols; j++){    // iterate through the columns of the second matrix
            matrix1.array[i][j] += matrix2.array[i][j];
        }
    }
}

// modifies the first vector to be the sum of the first and second vectors
void vector_plus_equal(Vector vector1, Vector vector2)
{
    // check that the dimensions of the matrices are the same
    if (vector1.length != vector2.length){
        // print out the and an error message
        printf("Vector 1: %d. ", vector1.length);
        printf("Vector 2: %d.\n", vector2.length);
        throw runtime_error("The dimensions of the vectors are not the same.");
    }

    // add the second vector to the first vector
    for (int i = 0; i < vector1.length; i++){        // iterate through the rows of the first matrix
        vector1.array[i] += vector2.array[i];
    }
}

double dot_product(Vector v1, Vector v2){
    // check that the vectors are the same size
    if (v1.length != v2.length){
        printf("vec1: %d, vec2: %d", v1.length, v2.length);
        throw runtime_error("The vectors are not the same size.");
    }

    // create a new vector to store the result of the multiplication
    double result = 0;

    // multiply the vectors
    for (int i=0; i<v1.length; i++){
        result += v1.array[i] * v2.array[i];
    }
    return result;
}


// returns a new vector object with the result of the subtraction
Vector subtract_vectors(Vector v1, Vector v2){
    // check that the vectors are the same size
    if (v1.length != v2.length){
        printf(" vec1: %d, vec2: %d", v1.length, v2.length);
        throw runtime_error("The vectors are not the same size.");
    }

    // create a new vector to store the result of the multiplication
    double *result = new double[v1.length];

    // multiply the vectors
    for (int i=0; i<v1.length; i++){
        result[i] = v1.array[i] - v2.array[i];
    }
    return Vector(result, v1.length);
}


Vector add_vectors(Vector v1, Vector v2){
    // check that the vectors are the same size
    if (v1.length != v2.length){
        printf("vec1: %d, vec2: %d", v1.length, v2.length);
        throw runtime_error("The vectors are not the same size.");
    }

    // create a new vector to store the result of the multiplication
    double *result = new double[v1.length];

    // multiply the vectors
    for (int i=0; i<v1.length; i++){
        result[i] = v1.array[i] + v2.array[i];
    }
    return Vector(result, v1.length);
}


// creates a new matrix object with a 2D array with the specified vector oriented as a row or column
Matrix vector_to_matrix(Vector v, bool horizontal=true){
    if (horizontal){
        double** arr = new double*[1];
        arr[0] = new double[v.length];
        for (int i=0; i<v.length; i++){
            arr[0][i] = v.array[i];
        }
        return Matrix(arr, 1, v.length);
    }
    else{
        double** arr = new double*[v.length];
        for (int i=0; i<v.length; i++){
            arr[i] = new double[1];
            arr[i][0] = v.array[i];
        }
        return Matrix(arr, v.length, 1);
    }
}


Vector matrix_to_vector(Matrix m){
    if (m.rows == 1 && m.cols >= 1){
        return Vector(m.array[0], m.cols);
    }
    else if (m.cols == 1 && m.rows > 1){
        double* arr = new double[m.rows];
        for (int i=0; i<m.rows; i++){
            arr[i] = m.array[i][0];
        }
        return Vector(arr, m.rows);
    }
    else{
        printf("The matrix is not a vector. It is %d x %d", m.rows, m.cols);
        return Vector();
    }
}


// multiply two one dimensional arrays with identical dimensions and return the result as a new Vector obj
Vector hadamard_product(Vector vec1, Vector vec2){
    // check that the dimensions of the matrices are the same
    if (vec1.length != vec2.length){
        // print out the and an error message
        printf("Vec 1: %d. ", vec1.length);
        printf("Vec 2: %d.\n", vec2.length);
        throw runtime_error("The dimensions of the vectors are not the same.\n");
    }

    double* result = new double[vec1.length];

    // iterate through the elements of the vectors
    for (int i = 0; i < vec1.length; i++){
        result[i] = vec1.array[i] * vec2.array[i];
    }
    return Vector(result, vec1.length);
}
    

// mutliplys two vectors together such that the result is a 2D matrix with element i,j = vec1[i] * vec2[j]
Matrix multiply_vectors_for_matrix(Vector vec1, Vector vec2){

    // printf("multiplying two vectors to make a matrix.\n");
    // vec1.print("v1");
    // vec2.print("v2");
    // create a new matrix to store the result of the subtraction
    double** result = new double*[vec1.length];

    // iterate through the rows of the first matrix
    for (int i = 0; i < vec1.length; i++){          // these will be rows
        result[i] = new double[vec2.length];
        for (int j = 0; j < vec2.length; j++){      // these will be cols
            result[i][j] = vec1.array[i] * vec2.array[j];
        }
    }
    // return a Array2DWithMetaData struct
    return Matrix(result, vec1.length, vec2.length);
}


// multiplies the matrix by the vector and returns the resulting vector such as: result = mat * vec
// assumes the vector is a column vector
Vector matrix_vector_multiply(Matrix mat, Vector vec){
     if (mat.cols == vec.length){
        // create a new matrix to store the result of the subtraction
        double* result = new double[mat.rows];

        // iterate through the rows of the first matrix
        for (int i = 0; i < mat.rows; i++){
            for (int j = 0; j < mat.cols; j++){
                result[i] += mat.array[i][j] * vec.array[j];
            }
        }
        return Vector(result, mat.rows);
    }
    else{
        printf("Matrix: %d x %d. ", mat.rows, mat.cols);
        printf("Vector: %d.\n", vec.length);
        throw runtime_error("The dimensions of the matrix and vector are not compatible.\n");
    }

}


// aggregate vectors into matrix assuming each vector is a row
Matrix aggregate_vectors(Vector* vecs, int n_vecs){
    // create a new matrix to store the result of the subtraction
    int length = vecs[0].length;
    double** result = new double*[n_vecs];

    // iterate through the rows of the first matrix
    for (int i = 0; i < n_vecs; i++){
        result[i] = new double[vecs[i].length];
        for (int j = 0; j < vecs[i].length; j++){
            result[i][j] = vecs[i].array[j];
        }
    }
    return Matrix(result, n_vecs, length);
}


double* random_1D_array_0_to_1_factory(int length){
    // create a new matrix with values randomized from 0 to 1 uniformly
    double *result = new double[length];

    // factory random numbers
    for (int i=0; i<length; i++){
        result[i] = (double)rand() / (double)RAND_MAX;
        // printf("made a new random number:[%d] = %f\n", i, result[i]);
    }
    // printf("\n");
    return result;
}


Vector random_vector_0_to_1_factory(int length){
    double* arr = random_1D_array_0_to_1_factory(length);
    return Vector(arr, length);
}


Matrix random_matrix_0_to_1_factory(int rows, int cols){
    // create a new matrix with values randomized from 0 to 1 uniformly. Returns a pointer to an array of pointers to the rows
    double** arr = new double*[rows];

    // factory random numbers
    for (int i=0; i<rows; i++){
        arr[i] = random_1D_array_0_to_1_factory(cols);
    }
    return Matrix(arr, rows, cols);
}


Matrix zeros_matrix_factory(int rows, int cols){

    // create a new matrix with values randomized from 0 to 1 uniformly. Returns a pointer to an array of pointers to the rows
    double** arr = new double*[rows];

    // factory random numbers
    for (int i=0; i<rows; i++){
        arr[i] = new double[cols];
        for (int j=0; j<cols; j++){
            arr[i][j] = 0;
        }
    }
    return Matrix(arr, rows, cols);
}


Vector zeros_vector_factory(int length){
    double* arr = new double[length];
    for (int i=0; i<length; i++){
        arr[i] = 0;
    }
    return Vector(arr, length);
}


// int main(){
//     // load from input weights
//     Matrix test = get_matrix_from_csv("model/input_weights.csv");
//     test.print("input weights");
// }