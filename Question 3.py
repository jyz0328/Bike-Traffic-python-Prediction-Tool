from functools import total_ordering
from tkinter import W
import pandas
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''
dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
# print(dataset_1.to_string()) #This line will print out your data
dataset_test = dataset_1

p_value = 0.05

#print(type(dataset_test))
#print(dataset_1.to_string())

#changing the data
dataset_test.replace(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], [1,2,3,4,5,6,7],inplace = True)

#print (dataset_test.to_string())

#Resort the table
dataset_test.sort_values("Day",inplace=True)

#print (dataset_test.to_string())

#Create sub table list for days
SubTableListForDays = []
for index in range(1, 8):
    SubTableListForDays.append(dataset_test[dataset_test['Day']==index])

#print(SubTableListForDays[0])
#print(SubTableListForDays[2]['Total'].values.tolist())

#Create sub table list for bridge and test
Result_table = [[0 for i in range(7)] for j in range(7)]
#print(Result_table)

for col in range(0,7):
    for row in range(0,7):
        stat, p = ttest_ind(SubTableListForDays[row]['Brooklyn Bridge'].values.tolist(), SubTableListForDays[col]['Brooklyn Bridge'].values.tolist())
#        stat, p = ttest_ind(SubTableListForDays[row]['Manhattan Bridge'].values.tolist(), SubTableListForDays[col]['Manhattan Bridge'].values.tolist())
#        stat, p = ttest_ind(SubTableListForDays[row]['Williamsburg Bridge'].values.tolist(), SubTableListForDays[col]['Williamsburg Bridge'].values.tolist())
#        stat, p = ttest_ind(SubTableListForDays[row]['Queensboro Bridge'].values.tolist(), SubTableListForDays[col]['Queensboro Bridge'].values.tolist())
        Result_table[row][col] = p

BoolResultTable = [[0 for i in range(7)] for j in range(7)]
for col in range(0,7):
    for row in range(0,7):
        BoolResultTable[row][col] = Result_table[row][col]<p_value

ColorTable = [[0 for i in range(7)] for j in range(7)]
for col in range(0,7):
    for row in range(0,7):
        if Result_table[row][col]<p_value:
            ColorTable[row][col] = 'g'
        else:
            ColorTable[row][col] = 'r'
#fig, ax = plt.subplots()       
#plt.patch.set_visible(False)
DaysList = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
plt.axis('off')
plt.axis('tight')
plt.title("days")
plt.table(cellText=BoolResultTable, loc='center',cellColours=ColorTable, rowLabels = DaysList, colLabels = DaysList)
plt.tight_layout()
plt.show()