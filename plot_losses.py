# read in the losses.txt file and plot the losses using matplotlib
import matplotlib.pyplot as plt

# read in the data
with open('losses.csv', 'r') as f:
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