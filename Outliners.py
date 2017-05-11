import pandas as pd
import matplotlib.pyplot as plt

def open_file(fileName):
    data = pd.read_csv(fileName)
    return data

def create_whisker_plot(data):
    print(data.size)
    data.plot(kind='box', subplots=True, layout=(3,13), sharex=False, sharey=False)
    plt.show()

def crete_history(data):
    data.hist()
    plt.show()

def delete_outliners(data, column, min_range, max_range):
    data_withoutOutliners = data[column].between(min_range, max_range, inclusive=True)
    show_data_info(data_withoutOutliners)

def show_data_info(data):
    print("Number of instance: " + str(data.shape[0])); #shape da la forma de la medida de los atributos y las instancias [0] -> numero de instancias
    # print("Number of fetures: " + str(data.shape[1]))  # shape da la forma de la medida de los atributos y las instancias [1] -> numero de campos
    print("----------------------------------------------------------------")
    print(data.head(10)) #Te muestra los titulos de cada una de las caracteristicas y un número de instancias
    print("Atributos numericos:")
    numerical_info = data.iloc[:, : data.shape[1]]
    print(numerical_info.describe())

def create_whisker_plot(data):
    print(data.size)
    data.plot(kind='box', subplots=True, layout=(3,13), sharex=False, sharey=False)
    plt.show()

if __name__ == '__main__':
    data = open_file("cleanTrain.csv")
    # Crea la grafica donde se ven los outliners
    create_whisker_plot(data)


    # delete_outliners(data, "OverallCond", 4, 7)

    # show_data_info(data)
    #data = get_feature_subset(data, "Survived", "Pclass", "Sex", "Embarked")
    #data = delete_colum(data, "PassengerId")

    #data = delete_missing_values(data, 'instance')
    #data = replace_missing_values_with_constant(data, 'Age', -1)
    #replace_missing_values_with_mean(data, 'Age')

    # print(data)