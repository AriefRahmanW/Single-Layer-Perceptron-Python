# Single Layer Perceptron Using Python Usage Guide

## Data Model

### dataset_batang.txt
total data : 75 <br />
total class : 3
- Batang-pisang = 25 data
- Batang-jambu = 25 data
- Batang-pepaya = 25 data


### dataset_iris.txt

total data : 150 <br />
total class : 3
- Iris-setosa = 50 data
- Iris-versicolor = 50 data
- Iris-virginia = 50 data

this dataset is gotten from https://archive.ics.uci.edu/ml/datasets/iris


## Get Started

install depedencies...

```bash
pip install -r requirements.txt
```
change this part as you wish...

```python
training = ConvertTxtToDataset("batang/training.txt").getSinglePerceptronDataset(
    y=(('Batang-pisang', 0.0), ('Batang-pepaya', 1.0)) # customize with datasete class
)

testing = ConvertTxtToDataset("batang/testing.txt").getSinglePerceptronDataset(
    y=(('Batang-pisang', 0.0), ('Batang-pepaya', 1.0)) # customize with datasete class
)
```
