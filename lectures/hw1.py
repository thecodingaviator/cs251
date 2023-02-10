import numpy as np

cow = np.array( [ [1.1,2.2,3.3], [2.1,4.2,6.2], [3.1,6.2,9.3], [4.1,8.2,12.3] ] ) # (4,3)

socks = np.mean( cow, axis = 0 ) # (3,)
print(socks.shape) # (1,3)

coat = cow.std( axis = 0 ) # (3,)
print(type(coat)) # <class 'numpy.ndarray'>

zed = (cow - socks) / coat # (4,3)

trend = np.mean( cow, axis=1 ) # (4,)

whatsit = np.mean( cow ) # ()

tree = np.random.random( (6,5,7) )

shrub = tree.mean( axis=0 ) # (5,7)


trimmed = tree - tree.mean(axis=0)

bush = tree.mean( axis = 2)

shears = tree.mean(axis=1)

# This line won't work without reshaping shears. 
# Write the fix-up line, then the size of trimmed.
shears = shears.reshape( (6,1,7) )

trimmed = tree - shears
print(f'trimmed is a {trimmed.shape}')