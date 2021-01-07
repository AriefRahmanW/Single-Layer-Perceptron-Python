# Single Layer Perceptron Using Python Usage Guide

## Get Started

install depedencies...

```bash
pip install -r requirements.txt
```

## Data Model

### dataset_batang.txt
total data : 75 <br />
total class : 3
- Batang-pisang = 25 data
- Batang-jambu = 25 data
- Batang-pepaya = 25 data

this dataset is gotten by using GLCM(texture analyst method)

0 deg | 45 deg | 90 deg | 135 deg | Class
----- | ------ | ------ | ------- | -----
11750114.13 | 11618426.39 | 18167705.75 | 11527839.21 | Batang-pepaya

### dataset_iris.txt
total data : 150 <br />
total class : 3
- Iris-setosa = 50 data
- Iris-versicolor = 50 data
- Iris-virginia = 50 data

this dataset is gotten from https://archive.ics.uci.edu/ml/datasets/iris

No | sepal length | sepal width | petal length | petal width | Class
-- | ------------ | ----------- | ------------ | ----------- | -----
1 | 5.1 | 3.5 | 1.4 | 0.2 | Iris-setosa

## Usage Description

```python
training, testing = ConvertTxtToDataset(
    filename="dataset_batang.txt"
).getSinglePerceptronDataset(
    x=(1,2), # input column
    compress=10000000, # compress dataset number if number is too large 
    y=(('Batang-pepaya', 0.0), ('Batang-pisang', 1.0)), # Output class, choose two class
    percentage=80 # saperate data into 80% training and 20% testing 
)
```



