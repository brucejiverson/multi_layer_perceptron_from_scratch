# run_cpp.py
# Script that compiles and executes a .cpp file
# Usage:
# python run_cpp.py -i <filename> (without .cpp extension)

import sys, os, getopt
import matplotlib.pyplot as plt


def main(argv):
    cpp_file = ''
    exe_file = ''
    try:
        opts, args = getopt.getopt(argv, "hi:",["help",'ifile='])
    except getopt.GetoptError as err:
        # print help information and exit
        print(err)      
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-i", "--ifile"):
            cpp_file = a + '.cpp'
            exe_file = a + '.exe'
            run(cpp_file, exe_file)


def usage():
    print('run_cpp.py -i <filename> (without .cpp extension)')

def run(cpp_file, exe_file):
    os.system("echo Compiling " + cpp_file)
    os.system('g++ ' + cpp_file + ' -o ' + exe_file)
    os.system("echo Running " + exe_file)
    os.system("echo -------------------")
    os.system("./" + exe_file)

if __name__=='__main__':
    main(sys.argv[1:])
    
    # read in the losses.txt file and plot the losses using matplotlib
    # read in the data
    with open('training_records/losses.csv', 'r') as f:
        data = f.read()
        
    # split the data into lines (comma separated) and ensure they are floats
    data = data.split(',')
    data = [float(x) for x in data if x != '']

    # now plot the data
    fig, ax = plt.subplots()
    plt.title('Losses during training')
    plt.scatter(range(len(data)), data)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.show()