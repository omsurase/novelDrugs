import hashlib
import numpy as np
from numpy import array, argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

completed_lines_hash = set()

#save processed data to smiles2.csv
new = open("smiles2.csv", "w")

#read data from smiles1.csv line by line
for line in open("smiles.csv", "r"):

    #remove duplicates
    if line not in completed_lines_hash:
        completed_lines_hash.add(line)
    # remove lines with less than 34 characters and more than 100 characters
        if 34 < len(line) < 100:
            line = line.rjust(len(line) + 1, "G")
            new.write(line)
new.close()



#read processed data from smiles2.csv
data = open ("smiles2.csv", "r").read()
#Create list of unique characters
chars = list (set(data))

#print(chars)
#TotalCharacters -> total number of characters in the smiles2.csv file
#vocab_size -> total number of unique characters in the smiles2.csv file
TotalCharacters, vocab_size = len (data), len (chars)
print ( 'data has %d characters, %d unique.'  % (TotalCharacters, vocab_size))
print ("Unique charaters are \n" + str(chars))



#array of unique characters
values = array(chars)
#print (values)
#unique numerical lables for each unique character between 0 and vocab_size-1
LabelEncoder = LabelEncoder()
num_encoded = LabelEncoder.fit_transform(values)
print(num_encoded)

#one-hot encoding of the unique characters to for a binary matrix of size [vocab_size, vocab_size]
OneHotEncoder = OneHotEncoder(sparse_output=False)
#reshaping the array to [vocab_size, 1] from [vocab_size]
num_encoded = num_encoded.reshape(len(num_encoded), 1)
#one-hot encoding the array
onehot_encoded = OneHotEncoder.fit_transform(num_encoded)
print(onehot_encoded)

#Read in processed data file
data = open("smiles2.csv", "r").read()
#Create a list of the dataset
datalist = list(data)
#print(datalist)
#Create an array of the dataset
dataarray = array(datalist)
#print (dataarray)
#Fit one-hot encoding to dataarray
dataarray = dataarray.reshape(len(dataarray), 1)
#print (dataarray)
OHESMILES = OneHotEncoder.fit_transform(dataarray).astype(int)
#print("Size of one-hot encoded array of data: " + str(OHESMILES.shape))
#print("One-hot encoded array of data:")
with open ("smiles3.csv", "w") as smiles_file:
    for row in OHESMILES:
        smiles_file.write(str(row))
#print(OHESMILES)
INTSMILES = [np.where(r==1)[0][0] for r in OHESMILES]
#print(INTSMILES)