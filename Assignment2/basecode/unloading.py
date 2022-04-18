import pickle

filename = 'params.pickle'

infile = open(filename,'rb')
new_dict = pickle.load(infile)
infile.close()

print(new_dict)