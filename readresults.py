import numpy as np

# Загрузите массив из файла .npy
arr = np.load('result_matvec.npy')
arr2 = np.load('result_cg_2d.npy')



# Теперь вы можете работать с массивом 'arr'
print(arr)
print('А теперь')
print(arr2)