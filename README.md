# Single Layer Perceptron Using Python Usage Guide

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

## Get Started

install depedencies...

```bash
pip install -r requirements.txt
```
change this part as you wish...

```python
[...]
training, testing = ConvertTxtToDataset(
    filename="dataset_batang.txt"
).getSinglePerceptronDataset(
    x=(1,2), # input column
    compress=10000000, # compress dataset number if number is too large 
    y=(('Batang-pepaya', 0.0), ('Batang-pisang', 1.0)), # Output class, choose two class
    percentage=80 # saperate data into 80% training and 20% testing 
)
[...]
```
training data

```
(1.0, 1.18, 1.82, 0.0)
(1.0, 3.76, 8.12, 0.0)
(1.0, 3.66, 8.04, 0.0)
(1.0, 1.22, 2.08, 0.0)
(1.0, 2.29, 4.03, 0.0)
(1.0, 2.71, 4.6, 0.0)
(1.0, 1.55, 3.6, 0.0)
(1.0, 3.03, 7.02, 0.0)
(1.0, 4.15, 7.29, 0.0)
(1.0, 0.99, 1.53, 0.0)
(1.0, 1.35, 1.86, 0.0)
(1.0, 1.09, 1.67, 0.0)
(1.0, 1.22, 2.07, 0.0)
(1.0, 1.26, 1.8, 0.0)
(1.0, 1.69, 3.18, 0.0)
(1.0, 2.14, 3.84, 0.0)
(1.0, 3.1, 6.94, 0.0)
(1.0, 2.15, 3.46, 0.0)
(1.0, 4.28, 6.38, 0.0)
(1.0, 1.69, 2.62, 0.0)
(1.0, 2.57, 2.71, 1.0)
(1.0, 3.46, 5.29, 1.0)
(1.0, 2.34, 1.83, 1.0)
(1.0, 4.31, 5.43, 1.0)
(1.0, 3.88, 5.15, 1.0)
(1.0, 4.11, 6.07, 1.0)
(1.0, 3.16, 3.49, 1.0)
(1.0, 3.6, 4.58, 1.0)
(1.0, 1.79, 1.81, 1.0)
(1.0, 2.21, 2.41, 1.0)
(1.0, 1.33, 1.39, 1.0)
(1.0, 1.49, 1.54, 1.0)
(1.0, 1.5, 1.42, 1.0)
(1.0, 2.16, 2.03, 1.0)
(1.0, 3.49, 4.38, 1.0)
(1.0, 1.77, 1.84, 1.0)
(1.0, 1.24, 1.53, 1.0)
(1.0, 1.99, 2.08, 1.0)
(1.0, 1.73, 2.13, 1.0)
(1.0, 1.89, 2.02, 1.0)
```