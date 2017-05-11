import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import scatter_matrix

def open_file(fileName):
    data = pd.read_csv(fileName)
    return data

def create_histogram(data):
    data.hist(column = 'bedrooms')

    plt.show()

def create_density_plot(data):
    data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
    plt.show()

def create_whisker_plots(data):
    data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
    plt.show()

def show_data_info(data):
    print("Number of instance: " + str(data.shape[0]))
    print("Number of fetures: " + str(data.shape[1]))

    print('------------------------------------------')

    print("Initial instances:\n")
    print(data.head(10))

    print("Numerical Information:\n")
    numerical_info = data.iloc[:, :data.shape[1]]
    print(numerical_info.describe())


def numero_cuartos_influye_precio(data):

    numRoms = data['TotRmsAbvGrd'].value_counts()
    numRomsKeys = numRoms.keys()
    print(numRomsKeys)

    priceArray = []
    for number in numRomsKeys:
        subset = data.loc[data['TotRmsAbvGrd'] == number]
        priceArray.append(subset["SalePrice"].mean())

    width = .5
    plt.bar(numRomsKeys, priceArray, width, color="blue")

    plt.ylabel('precio')
    plt.xlabel('#Cuartos')
    plt.title('Cuartos inlfuye precio')
    plt.xticks(np.arange(min(numRomsKeys), max(numRomsKeys), 1))
    plt.yticks(np.arange(0, max(priceArray), 20000))
    plt.show()

def cantidad_carros_influye_Precio(data):

    numCars = data['GarageCars'].value_counts()
    numCarsKeys = numCars.keys()

    priceArray = []
    for number in numCarsKeys:
        subset = data.loc[data['GarageCars'] == number]
        priceArray.append(subset["SalePrice"].mean())

    width = .5
    plt.bar(numCarsKeys, priceArray, width, color="blue")

    plt.ylabel('Precio')
    plt.xlabel('Cantidad de vehiculos por garaje')
    plt.title('Cantidad de carros inlfuye en el precio')
    plt.xticks(np.arange(min(numCarsKeys), max(numCarsKeys), 1))
    plt.yticks(np.arange(0, max(priceArray), 20000))
    plt.show()

def vecindario_influye_precio(data):
    neighborhoods = data['Neighborhood'].value_counts()
    neighborhoodsKeys = neighborhoods.keys()
    priceArray = []
    keyArray=[]
    for key in neighborhoodsKeys:
        subset = data.loc[data['Neighborhood'] == key]
        keyArray.append(str(key))
        priceArray.append(subset["SalePrice"].mean())
    plt.bar(np.arange(len(priceArray)), priceArray, color="blue")

    plt.ylabel('precio')
    plt.xlabel('#Vecindario')
    plt.title('Vecindario influye precio')
    plt.xticks(np.arange(0, len(keyArray)), keyArray)
    plt.yticks(np.arange(0, max(priceArray), 20000))
    plt.show()

def calidad_piscina_influye_precio(data):
    typesPiscina = data['PoolQC'].value_counts()
    typesPiscinasKeys = typesPiscina.keys()
    priceArray = []
    keyArray=[]
    for key in typesPiscinasKeys:
        subset = data.loc[data['PoolQC'] == key]
        keyArray.append(str(key))
        priceArray.append(subset["SalePrice"].mean())
    plt.bar(np.arange(len(priceArray)), priceArray, color="blue")

    plt.ylabel('Precio')
    plt.xlabel('Tipo de Piscina')
    plt.title('Tipo de piscina influye precio')
    plt.xticks(np.arange(0, len(keyArray)), keyArray)
    plt.yticks(np.arange(0, max(priceArray), 20000))
    plt.show()

def calidad_cerca_influye_precio(data):
    typesCerca = data['Fence'].value_counts()
    typesCercaKeys = typesCerca.keys()
    priceArray = []
    keyArray=[]
    for key in typesCercaKeys:
        subset = data.loc[data['Fence'] == key]
        keyArray.append(str(key))
        priceArray.append(subset["SalePrice"].mean())
    plt.bar(np.arange(len(priceArray)), priceArray, color="blue")

    plt.ylabel('Precio')
    plt.xlabel('Tipo de Cerca')
    plt.title('Tipo de cerca influye precio')
    plt.xticks(np.arange(0, len(keyArray)), keyArray)
    plt.yticks(np.arange(0, max(priceArray), 20000))
    plt.show()

def calidad_cocina_influye_precio(data):
    typesCocina = data['KitchenQual'].value_counts()
    typesCocinaKeys = typesCocina.keys()
    priceArray = []
    keyArray=[]
    for key in typesCocinaKeys:
        subset = data.loc[data['KitchenQual'] == key]
        keyArray.append(str(key))
        priceArray.append(subset["SalePrice"].mean())
    plt.bar(np.arange(len(priceArray)), priceArray, color="blue")

    plt.ylabel('Precio')
    plt.xlabel('Calidad de la cocina')
    plt.title('Calidad de Cocina influye en el precio')
    plt.xticks(np.arange(0, len(keyArray)), keyArray)
    plt.yticks(np.arange(0, max(priceArray), 20000))
    plt.show()

def configuracion_lote_influye_precio(data):
    typesLote = data['LotConfig'].value_counts()
    typesLoteKeys = typesLote.keys()
    priceArray = []
    keyArray=[]
    for key in typesLoteKeys:
        subset = data.loc[data['LotConfig'] == key]
        keyArray.append(str(key))
        priceArray.append(subset["SalePrice"].mean())
    plt.bar(np.arange(len(priceArray)), priceArray, color="blue")

    plt.ylabel('Precio')
    plt.xlabel('Configuración del Lote')
    plt.title('Configuración del Lote influye en el precio')
    plt.xticks(np.arange(0, len(keyArray)), keyArray)
    plt.yticks(np.arange(0, max(priceArray), 20000))
    plt.show()

def utilidades_influye_precio(data):
    typesLote = data['Utilities'].value_counts()
    typesLoteKeys = typesLote.keys()
    priceArray = []
    keyArray=[]
    for key in typesLoteKeys:
        subset = data.loc[data['Utilities'] == key]
        keyArray.append(str(key))
        priceArray.append(subset["SalePrice"].mean())
    plt.bar(np.arange(len(priceArray)), priceArray, color="blue")

    plt.ylabel('Precio')
    plt.xlabel('Servicios incluidos')
    plt.title('Servicios que incluyen influye en el precio')
    plt.xticks(np.arange(0, len(keyArray)), keyArray)
    plt.yticks(np.arange(0, max(priceArray), 20000))
    plt.show()

def street_influye_precio(data):
    typesStreets = data['Street'].value_counts()
    typesStreetsKeys = typesStreets.keys()
    priceArray = []
    keyArray=[]
    for key in typesStreetsKeys:
        subset = data.loc[data['Street'] == key]
        keyArray.append(str(key))
        priceArray.append(subset["SalePrice"].mean())
    plt.bar(np.arange(len(priceArray)), priceArray, color="blue")

    plt.ylabel('Precio')
    plt.xlabel('Tipos de calle')
    plt.title('Tipos de calle influye en el precio')
    plt.xticks(np.arange(0, len(keyArray)), keyArray)
    plt.yticks(np.arange(0, max(priceArray), 20000))
    plt.show()

def garaje_influye_precio(data):
    typesGaraje = data['GarageType'].value_counts()
    typesGarajeKeys = typesGaraje.keys()
    priceArray = []
    keyArray=[]
    for key in typesGarajeKeys:
        subset = data.loc[data['GarageType'] == key]
        keyArray.append(str(key))
        priceArray.append(subset["SalePrice"].mean())
    plt.bar(np.arange(len(priceArray)), priceArray, color="blue")

    plt.ylabel('Precio')
    plt.xlabel('Tipos de Garaje')
    plt.title('Tipos de Garaje influye en el precio')
    plt.xticks(np.arange(0, len(keyArray)), keyArray)
    plt.yticks(np.arange(0, max(priceArray), 20000))
    plt.show()

if __name__ == '__main__':
    filePath = "train.csv"

    data = open_file(filePath)

    # vecindario_influye_precio(data)
    # numero_cuartos_influye_precio(data)
    # cantidad_carros_influye_Precio(data)
    # calidad_piscina_influye_precio(data)
    # calidad_cerca_influye_precio(data)
    # calidad_cocina_influye_precio(data)
    # configuracion_lote_influye_precio(data)
    # utilidades_influye_precio(data)
    # street_influye_precio(data)
    garaje_influye_precio(data)