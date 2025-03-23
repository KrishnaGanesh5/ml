import random 
import csv 
attributes=[["Sunny","Rainy"],["Warm","Cold"],["Normal","High"],["Strong","Weak"],["Warm","Cool"],["Same","Change"]] 
num_attributes=len(attributes) 
print("The most general hypothesis: ['?','?','?','?','?','?']") 
print("The most general hypothesis: ['0','0','0','0','0','0']") 
a=[] 
print("The given training dataset: ") 
with open('/content/Week-1.csv','r') as csvFile: 
  reader=csv.reader(csvFile) 
  for row in reader: 
    a.append(row) 
    print(row) 
print("The initial value of hypothesis: ") 
hypothesis=['0']*num_attributes 
print(hypothesis) 
for j in range(0,num_attributes): 
  hypothesis[j]=a[0][j] 
print("FIND-S: Finding a Maximality Specific Hypothesis") 
for i in range(0,len(a)): 
  if a[i][num_attributes]=="yes": 
    for j in range(0,num_attributes): 
      if a[i][j]!=hypothesis[j]: 
        hypothesis[j]='?' 
      else: 
        hypothesis[j]=a[i][j] 
  print("For training example no: {0} the hypothesis is ".format(i),hypothesis) 
print("The Maximally Specific Hypothesis for a given training examples:") 
print(hypothesis) 