#include "linear_algebra.cpp"

// a class that holds a 2D matrix of doubles and a vector of strings that are the column names
class DataFrame {
    public:
        Matrix data;
        std::vector<std::string> column_names;
        DataFrame(Matrix data, std::vector<std::string> column_names){
            this->data = data;
            this->column_names = column_names;
        }

        // print the first n rows of the dataframe
        void head(int n = 5){
            if (n > this->data.n_rows){
                n = this->data.n_rows;
            }

            // print the column names
            for (int i = 0; i < this->column_names.size(); i++){
                printf("%s", this->column_names[i].c_str());
                if (i != this->column_names.size()-1){
                    printf(", ");
                }
            }
            printf("\n");
            print_2D_array(this->data.array, n, this->data.n_cols);
        }
        
        void tail(int n = 5){
            if (n > this->data.n_rows){
                n = this->data.n_rows;
            }

            // print the column names
            for (int i = 0; i < this->column_names.size(); i++){
                printf("%s", this->column_names[i].c_str());
                if (i != this->column_names.size()-1){
                    printf(", ");
                }
            }
            printf("\n");
            print_2D_array(this->data.array + this->data.n_rows - n, n, this->data.n_cols);
        }

        Vector get_column(std::string col_name){
            int col = 0;
            for (int i = 0; i < this->column_names.size(); i++){
                if (this->column_names[i] == col_name){
                    col = i;
                    break;
                }
            }
            return data.get_column(col);
        }

        Matrix get_subset_by_columns(std::vector<std::string> col_names){
            int n_cols = col_names.size();

            // get the column indices
            int* col_indices = new int[n_cols];
            for (int i = 0; i < n_cols; i++){
                for (int j = 0; j < this->column_names.size(); j++){
                    if (this->column_names[j] == col_names[i]){
                        col_indices[i] = j;
                        break;
                    }
                }
            }

            double** subset = new double*[this->data.n_rows];
            for (int i = 0; i < this->data.n_rows; i++){
                subset[i] = new double[n_cols];
                for (int j = 0; j < n_cols; j++){
                    subset[i][j] = this->data.array[i][col_indices[j]];
                }
            }
            return Matrix(subset, this->data.n_rows, n_cols);
        }
};


DataFrame load_dataframe_from_csv(std::string filename){
    ifstream file(filename);        // open the file
    string line;                    // to hold each line
    vector<string> lines;           // vector to hold the lines

    // read the file line by line
    while (getline(file, line)){
        lines.push_back(line);
    }

    // get the number of rows and columns
    int n_rows = lines.size();        // note that this counts the header as a row
    int n_cols = 0;
    for (int i = 0; i < lines[0].size(); i++){
        if (lines[0][i] == ','){            
            n_cols++;
        }
    }

    if (lines[0][lines[0].size()-1] == ',' && lines[0][lines[0].size()-2] == ','){
        n_cols--;
    }

    // get the column names in order from the first line
    std::vector<std::string> column_names;
    double** data = new double*[n_rows];

    // create a 2D array to hold the data
    for (int i = 0; i < n_rows; i++){
        data[i] = new double[n_cols];
    }

    // fill the data
    for (int i = 0; i < n_rows; i++){
        int col = 0;
        string num = "";
        for (int j = 0; j < lines[i].size(); j++){
            if (lines[i][j] == ',' && num != ""){
                if (i == 0){
                    column_names.push_back(num);
                }
                else{
                    data[i][col] = stod(num);
                }
                num = "";
                col++;
            }
            else{
                num += lines[i][j]; // add the character to the string representing the number
            }
        }
    }
    return DataFrame(Matrix(data, n_rows-1, n_cols), column_names);
}


int main(){
    // DataFrame df = load_dataframe_from_csv("data/winequalityN.csv");
    DataFrame df = load_dataframe_from_csv("data/W1data.csv");
    df.head();
    return 0;
}