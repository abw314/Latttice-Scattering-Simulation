import numpy as np

## n, m define lattice size
## lattice size vars define the spacing in the x and y directions of the lattice
n = 3
m = 3
latticeSizeX = 2
latticeSizeY = 2

#creating the lattice
lattice = np.zeros((n, m), dtype=[('position', float, 2), ('size', float, 1)])

# initializing the lattice postions
for i in range(n):
    for j in range(m):
        lattice['position'][i][j] = [latticeSizeX*(i+1), latticeSizeY*(j+1)]

print(lattice)
print('the lattice column is ', lattice[1]['position'])
print('the lattice position is', lattice[1][1]['position'])
