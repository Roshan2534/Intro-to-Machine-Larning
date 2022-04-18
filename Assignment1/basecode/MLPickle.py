import pickle
import csv

filename = 'diabetes.pickle'

f = open(filename, 'rb')
X,y,Xtest,ytest = pickle.load(f, encoding='latin1')


with open('Diabetes_data.csv', 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter1 = csv.Sniffer.has_header(file)