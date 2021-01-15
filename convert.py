class ConvertTxtToDataset:

    def __init__(
        self,
        filename=""
        ):

        read_data = open(filename, "r").read()

        split_data = read_data.split("\n")

        self.arr_data = []
        for data in split_data:
            self.arr_data.append([float(i) for i in data.split("\t")[:-1]] + [data.split("\t")[-1]])

    def getAllDataset(self):
        return self.arr_data

    def getSinglePerceptronDataset( # mengubah dataset agar sesuai format single layer perceptron
        self,
        compress=1,
        y=(('Batang-pisang', 0.0), ('Batang-jambu', 1.0))
        ):

        converted_data = []

        if compress == 0: # secara default compress bernilai 0

            converted_data = [[1.0] + i for i in self.arr_data]
        else:
            converted_data = [[1.0] + i for i in self.arr_data]

        dataset = []

        for i in converted_data:
            for j in range(len(y)):
                if i[-1] == y[j][0]:
                    dataset.append( i[:-1] + [y[j][1]] )

        return dataset


