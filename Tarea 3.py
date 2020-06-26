#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Tarea 3 Fabian Zamora C B57983 Modelos probabilisticos de señales y sistemas

import numpy as np 
import pandas as pd
import csv
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

#Lectura de los archivos proporcionados
xy = pd.read_csv('xy.csv', names=['5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'],header =0);
xyp = pd.read_csv('xyp.csv',names = ['x','y','p'],header = 0)


#Cuarta pregunta 

x_graph = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] #Vector de los valores de X
filas = xy.sum(axis=1) #Valores para la probabilidad marginal X
plt.figure(1)
plt.plot(x_graph,filas)
plt.xlabel('Valores de X') 
plt.ylabel('Suma de las filas')   
plt.title('Funcion de densidad marginal X') 
plt.show() #Funcion de densidad marginal X a partir de los datos


y_graph = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]#Vector de los valores de Y
columnas = xy.sum(axis=0) #Valores para la probabilidad marginal Y
plt.figure(0)
plt.plot(y_graph,columnas)
plt.xlabel('Valores de Y') 
plt.ylabel('Suma de las columnas')   
plt.title('Funcion de densidad marginal Y')  
plt.show() #Funcion de densidad marginal Y a partir de los datos


#Primera pregunta


def gauss(x , mu, sigma): #Se definen los parametros mu y sigma (son los que buscamos ajustar)
    return 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-x-mu)**2/(2*sigma**2) #Funcion para la de densidad de probabilidad Gaussiana

x_1 = np.linspace(5,15,20)  #Vector de valores X
y_1 = np.linspace(5,25, 20) #Vector de valores de Y
   
u_x = sum(xyp.x * xyp.p) #Mu de X segun los datos
s_x = np.sqrt(sum(xyp.p*xyp.x**2)-(u_x)**2) #Sigma de Y segun los datos

u_y = sum(xyp.y * xyp.p) #Mu de Y segun los datos
s_y = np.sqrt(sum(xyp.p*xyp.y**2)-(u_y)**2) #Sigma de Y segun los datos


print('A partir de los datos el Mu para la funcion de densidad marginal X: '+str(u_x))
print('A partir de los datos el Sigma para la funcion de densidad marginal X: '+str(s_x))

print('A partir de los datos el Mu para la funcion de densidad marginal Y: '+str(u_y))
print('A partir de los datos el Sigma para la funcion de densidad marginal Y: '+str(s_y))

#Se usaran los valores reales para comparar el modelo con uno ideal 

k_x = gauss(x_1,0,1)    #Modelo ideal para X
r_x = gauss(x_1,u_x,s_x)#Modelo real para X
k_y = gauss(y_1,0,1)    #Modelo ideal para Y
r_y = gauss(y_1,u_y,s_y)#Modelo real para Y


#Se encuentran los parametros necesarios para la validez del modelo

paramx, _ = curve_fit(gauss,r_x, k_x)
paramy, _ = curve_fit(gauss,r_y, k_y)


print('Los parametros de ajuste para la funcion de densidad marginal X son (Mu y Sigma respectivamente): ' + str(paramx))
print('Los parametros de ajuste para la funcion de densidad marginal Y son (Mu y Sigma respectivamente): ' + str(paramy))



#Segunda pregunta

#La pregunta 2 se responde en el README, a grandes rasgos, consiste en la multiplcación de las funciones


#Tercera pregunta

#Los valores se hayaron por medio de las fórmulas vistas, su significado se expresan en el README

correlacion = sum(xyp.x*xyp.y*xyp.p)

print('La correlacion es: '+ str(correlacion))


#Se calculan las medias para la covarianza
meanx = xyp['x'].mean() 
meany = xyp['y'].mean()
covarianza = sum((xyp.x - meanx)*(xyp.y - meany)*xyp.p)

print('La covarianza es: '+ str(covarianza))


#Se calculas las desviaciones estandar para el coeficiente de Pearson
desvx = xyp['x'].std()
desvy = xyp['y'].std()
pearson = covarianza/(desvx * desvy)

print('El coeficiente de Pearson es: '+ str(pearson))


# In[ ]:





# In[ ]:




