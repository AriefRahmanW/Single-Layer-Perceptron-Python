class ConvertTxtToDataset:

    def __init__(
        self,
        filename="dataset_iris.txt"
        ):

        a = open(filename, "r").read()

        self.arr_data = a.split("\n")

        

    def getSinglePerceptronDataset( # mengubah dataset agar sesuai format single layer perceptron
        self,
        x=(0,1),
        compress=0,
        y=(('Iris-setosa', 0.0), ('Iris-versicolor', 1.0)),
        percentage=70
        ):

        if compress == 0: # secara default compress bernilai 0

            converted_data = [
                (
                    1.0,
                    float(i.split("\t")[ x[0] ]),
                    float(i.split("\t")[ x[1] ]),
                    i.split("\t")[-1]

                ) for i in self.arr_data ]
        else:
            converted_data = [
                (
                    1.0,
                    round(float(i.split("\t")[ x[0] ])/compress, 2),
                    round(float(i.split("\t")[ x[1] ])/compress, 2),
                    i.split("\t")[-1]

                ) for i in self.arr_data ]

        dataset = {}

        for i in range(len(y)):
            dataset [ y[i][0] ] = []

        for i in converted_data:
            for j in range(len(y)):
                if i[-1] == y[j][0]:
                    dataset [ y[j][0] ].append( (i[0], i[1], i[2], y[j][1]) )

        training = []
        testing = []

        # saparate data into traning and testing

        for i in range(len(y)):
            training += dataset[ y[i][0] ][: int(len(dataset[ y[i][0] ]) * percentage / 100)]
            testing += dataset[ y[i][0] ][int(len(dataset[ y[i][0] ]) * percentage / 100):]

        return training, testing

    def getAllDataset(self):
        return self.arr_data

# contoh penggunaan

#membagi data menjadi training dan testing
traning, testing = ConvertTxtToDataset(
    filename="dataset_batang.txt"
).getSinglePerceptronDataset(
    x=(0,2), # kolom yang dipilih untuk di training
    compress=10000000, # kompress berguna jika angka pada datasetnya terlalu besar
    y=(('Batang-pepaya', 0.0), ('Batang-pisang', 1.0)), # class output dari perceptron sesuaikan dengan kolom akhir dari file txt
    percentage=80 # mengambil 80% sebagai training dan 20% testing
)

for i in traning:
    print(i)
