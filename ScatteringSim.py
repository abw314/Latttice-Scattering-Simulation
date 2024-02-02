# 2d wavepacket test
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import filters
from matplotlib import animation
import os

# lattice potential
# requires n, m to be odd. else it doesn't compile. 
def Vgen(n,m,nx,ny,lx,ly):
    
    ## GRIDPOINT GRID WITHIN UNIT CELL
    # Assign unit cell, gridpoint parameters
    hx = lx / nx
    hy = ly / ny

    # Create an array of gridpoints within one unit cell
    R = np.zeros([2,nx,ny])
    R[0,:,:] = hx / 2
    R[1,:,:] = hy / 2

    # Assign proper XY coordinates
    for i in range(0,nx):
        for j in range(0,ny):
            R[0,i,j] += i * hx
            R[1,i,j] += j * hy

    ## PERIODIC APPROXIMATION GRID
    # Specify furthest extent desired
    N = 4

    # Create array of gridpoints around central unit cell
    Rs = np.zeros([2,2 * N - 1, 2 * N - 1])

    # Iterate over every point, assign correct XY coordinate
    for i in range(0,2*N - 1):

        for j in range(0,2*N - 1):

            Rs[:,i,j] = np.array([(i - N + 1),(j - N + 1)])

# Iterate over every point, assign correct XY coordinate
    for i in range(0,2*N - 1):

        for j in range(0,2*N - 1):

            Rs[:,i,j] = np.array([lx * (i - N + 1),ly * (j - N + 1)])

    # Reshape into a list of coordinates
    Rs = Rs.reshape(2,(2 * N - 1)**2)
    R = R.reshape(2,nx * ny)
    V = np.zeros(nx * ny)

    for r in range(0,nx*ny):

        v = 0

        for rs in range(0,(2 * N - 1)**2):

            v += 30 / np.linalg.norm(R[:,r] - Rs[:,rs])

        V[r] = v

    #Output V
    V = V.reshape([nx,ny])
    V = V - V.min()
    
    #Tile V such that it matches the shape of the lattice in the simulation
    V1 = V

    for i in range(0,m - 1):
        V1 = np.concatenate((V1,V),axis=0)

    V2 = V1

    for i in range(0,n - 1):
        V2 = np.concatenate((V2,V1),axis=1)

    return V2

# gaussian potential
# requires nx, ny to be even else it fails to compile. 
def V_simple(nx,ny,sigma,mu,r):
    V = np.zeros([nx,ny])
    cutoffx = nx
    cutoffy = ny
    midx = int(nx / 2)
    midy = int(ny / 2)
    r0 = np.array([midx,midy])
    xx = np.arange(midx - cutoffx,midx + cutoffx)
    yy = np.arange(midy - cutoffy,midy + cutoffy)

    for i in range(0,cutoffx):
        for j in range(0,cutoffy):
            V[i + midx - cutoffx,j + midy - cutoffy] = mu * np.exp(-((xx[i]-r[0])**2 + (yy[j]-r[1])**2)/(2 * (sigma)**2))

    V[:,midy:] = np.flip(V[:,:midy],axis=1)
    V[midx:,:] = np.flip(V[:midx,:],axis=0)
    return V

## n, m define lattice size
## here it is a 3 x 3 lattice array
## lattice size vars define the spacing in the x and y directions of the lattice
n = 9
m = 9
latticeSizeX = 6
latticeSizeY = 6

#creating the lattice
#each point has a location and a charge with some generic (and highly unreasonble) value of 1 Coulomb. 
#may need to change this later dependig on how the potential function is defined. 
lattice = np.zeros((n, m), dtype=[('position', float, 2), ('charge', float, 1)])

# initializing the lattice postions
for i in range(n):
    for j in range(m):
        lattice['position'][j][i] = [latticeSizeX*(i+1), latticeSizeY*(j+1)]

# creating time interval in (seconds?)
timeInterval = (0,10)
t = np.linspace(timeInterval[0],timeInterval[1],1000)

# creating spatial grid & grid spacing var
# may add finer resolution later. as it stands, this is enough for testing
xlim = (0,10)
xpts = 99
ylim = (0,10)
ypts = 99
x = np.linspace(xlim[0],xlim[1],xpts)
y = np.linspace(ylim[0],ylim[1],ypts)
xx, yy = np.meshgrid(x,y)
da = x[1]-x[2]

# Define some arbitrary, unitless gaussian wave packet, and plot just to make sure
# ideally all the directions and origins of the wave packets would be random. 
# we will plot them in a ordered manner just to see the beautiful animations. 
# also, we want to test out if our program will show wave interference like we know it will irl

mu = np.sqrt(np.pi/2) # normalization constant
r0 = (3,5) # center of the lattice
r1 = (2,5) # origin of wave packet 1
r2 = (5,2) # origin of wave packet 2
r3 = (8,5) # origin of wave packet 3
r4 = (5,8) # origin of wave packet 4
sigma = 0.1
delta = 10
eta = 1000
'''
k1 = np.array([10,0]) # direction of wave packet 1  
k2 = np.array([0,10]) # direction of wave packet 2
k3 = np.array([-10,0]) # direction of wave packet 3
k4 = np.array([0,-10]) # direction of wave packet 4
'''


# using the inverse of the lattice spacing should be an en example of a forbidden band.
k1 = np.array([1/latticeSizeX,0]) # direction of wave packet 1  
k2 = np.array([0,1/latticeSizeY]) # direction of wave packet 2
k3 = np.array([-1/latticeSizeX,0]) # direction of wave packet 3
k4 = np.array([0,-1/latticeSizeY]) # direction of wave packet 4


# since psi is defined everywhere in 2d space, it needs to be a 2d array. 
def psi0(xx, yy, r, k):
	return np.exp(-((xx-r[0])**2 + (yy-r[1])**2)/(2 * (sigma)**2)) * np.exp(k[0] * 1j * xx + k[1] * 1j * yy) / mu

# defining psi as the 2d array and then giving it the initial condition 
# with a wave packet at some starting point in space. 
psi = np.zeros((xpts, ypts), dtype = 'complex')
for i in range(len(x)):
	for j in range(len(y)):
		psi[j][i] = psi0(x[i], y[j], r1, k1) + psi0(x[i], y[j], r2, k2) + psi0(x[i], y[j], r3, k3) + psi0(x[i], y[j], r4, k4)

def laplace(array):
	global x
	M = x.size - 1
	laplacian = np.zeros((M+1,M+1),dtype='complex')
	dx2 = array[2:,1:M] + array[0:M-1,1:M] - 2*array[1:M,1:M]
	dy2 = array[1:M,2:] + array[1:M,0:M-1] - 2*array[1:M,1:M]
	laplacian[1:M,1:M] = (1/(da**2)) * (dx2 + dy2)

	return laplacian

# Define a function f to solve via the Runge-Kutta Method of the 4th order
# dpsi/dt = f(psi(x)) = (laplacian(psi(x,t)) - V(x)psi(x,t)) * i
def f(psi):
    
	lap = laplace(psi)

	return (lap - V * psi) * 1j

# Calculate psi(x, y, t + dt) from psi(x, y, t) by using the Runge-Kutta Method of the 4th order
# This function takes in the current wave function and evolves it a specific time interval
def timestep():
	global psi
	# Set time interval; this decision is very finnicky
	dt = 0.001

	# Stages of fourth order Runge Kutta Method
	k1 = f(psi) 
	k2 = f(psi + (dt/2) * k1) ; k3 = f(psi + (dt/2) * k2)
	k4 = f(psi + dt * k3)
	psi = psi + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

	psi[0,:] = 0
	psi[-1,:] = 0
	psi[:,0] = 0
	psi[:,-1] = 0 ; psi = psi / np.absolute(psi).sum()
    
	return


# Calculate the potential field caused by lattice points
V = Vgen(n,m,int(xpts/n),int(ypts/m),latticeSizeX,latticeSizeY)

# potential for a free particle. 
# V = np.zeros((xpts,ypts))

# potential with gaussian circle in the middle of the lattice
# V = V_simple(xpts, ypts, delta, eta, r0)

# building all the psi arrays over our time interval
probAmplitude = [psi]
probSpace = []
for i in range(len(t)):
	timestep()
	probAmplitude.append(psi)

# building the probability arrays
probSpace = []
for i in range(len(probAmplitude)):
	probSpace.append(abs(probAmplitude[i])**2)

# Plotting and animation.  
fig = plt.figure(figsize=(7, 7))

'''
def animate(i):
	global x, y, probAmplitude
	cont = plt.contourf(x, y, probSpace[i], i)
	return cont

anim = animation.FuncAnimation(fig, animate, frames=len(probAmplitude), interval=20, blit=False)
'''
'''
# saving the animation to a .mp4 file. 
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='UGQM'), bitrate=1800)
anim.save('im.mp4', writer=writer)
'''

# saving all the animation images. might be faster to do this
# and then turn it into a gif later. saving the animation using
# the writer above, is just too slow. takes hours to save a few 
# seconds of video. 
for i in range(len(probSpace)):
    plt.contourf(x, y, probSpace[i])
    fileName = "Forbidden" + str(i)
    plt.savefig(fileName)

plt.show()
