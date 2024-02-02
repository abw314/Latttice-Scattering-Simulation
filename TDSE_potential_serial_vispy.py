# 2d wavepacket test
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import filters
from matplotlib import animation
import vispy.plot as vplot
from vispy import app, scene
import multiprocessing as mp
import imageio
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

            Rs[:,i,j] = np.array([lx * (i - N + 1),ly * (j - N + 1)])

    # Reshape into a list of coordinates
    Rs = Rs.reshape(2,(2 * N - 1)**2)
    R = R.reshape(2,nx * ny)
    V = np.zeros(nx * ny)

    for r in range(0,nx*ny):

        v = 0

        for rs in range(0,(2 * N - 1)**2):

            v += 1 / np.linalg.norm(R[:,r] - Rs[:,rs])

        V[r] = v

    #Output V
    V = V.reshape([nx,ny])
    V = V - V.min() + 1

    #Tile V such that it matches the shape of the lattice in the simulation
    V1 = V

    for i in range(0,m - 1):
        V1 = np.concatenate((V1,V),axis=0)

    V2 = V1

    for i in range(0,n - 1):
        V2 = np.concatenate((V2,V1),axis=1)
    
    return V2

## n, m define lattice size
## here it is a 3 x 3 lattice array
## lattice size vars define the spacing in the x and y directions of the lattice
n = 3
m = 3
latticeSizeX = 2
latticeSizeY = 2

#creating the lattice
#each point has a size and a charge with some generic (and highly unreasonble) value of 1 Coulomb. 
#may need to change this later dependig on how the potential function is defined. 
lattice = np.zeros((n, m), dtype=[('position', float, 2), ('charge', float, 1)])

# initializing the lattice postions
for i in range(n):
    for j in range(m):
        lattice['position'][i][j] = [latticeSizeX*(i+1), latticeSizeY*(j+1)]

"""
print(lattice)
print('the lattice column is ', lattice[1]['position'])
print('the lattice position is', lattice[1][1]['position'])
"""

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

# Calculate the potential field caused by lattice points
V = Vgen(n,m,int(xpts/n),int(ypts/m),latticeSizeX,latticeSizeY)

# Define some arbitrary, unitless gaussian wave packet, and plot just to make sure
mu = 1
r0 = (2,5)
sigma = 0.1
k = np.array([20,0])

# since psi is defined everywhere in 2d space, it needs to be a 2d array. 
def psi0(xx, yy):
	return mu * np.exp(-((xx-r0[0])**2 + (yy-r0[1])**2)/(2 * (sigma)**2)) * np.exp(k[0] * 1j * xx + k[1] * 1j * yy)

# defining psi as the 2d array and then giving it the initial condition 
# with a wave packet at some starting point in space. 
psi = np.zeros((xpts, ypts), dtype = 'complex')
for i in range(len(x)):
	for j in range(len(y)):
		psi[i][j] = psi0(x[j], y[i])
        

'''
need to set boundary conditions on psi. it must be normalizable and I am going to assume the 
lattice is just the entire material.  we should trap the electrons inside the material 
and allow them to bounce around. so we can set the wave function to zero at the boundaries.
psi[0] = psi[xpts - 1] = 0
'''

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
# leading constants needs to be fixed. 
def f(psi):
    
	lap = laplace(psi)

	return (lap - V * psi) * 1j

# Calculate psi(x, t + dt) from psi(x,t) by using the Runge-Kutta Method of the 4th order
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

# building all the psi arrays over our time interval
"""def blockedTimestep(rank, tCnt)
    #decide 
    start = 0
    stop = 0
    if rank < n % p:
    else:"""

probAmplitude = [psi]
for i in range(len(t)):
    timestep()
    probAmplitude.append(psi)

probSpace = np.empty((t.shape[0] + 1, xpts, ypts))
vizProbs = np.empty((t.shape[0] + 1, xpts, ypts, 3), dtype="uint8")
for i in range(len(probAmplitude)):
    probSpace[i] = abs(probAmplitude[i])**2
    norm = cv2.normalize(probSpace[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    vizProbs[i] = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    vizProbs[i] = cv2.cvtColor(vizProbs[i], cv2.COLOR_BGR2RGB)

#vizProbs = probSpace[:] / np.max(probSpace[:])

imageio.mimwrite("scatterSim.mp4", vizProbs, fps=30)

# Plotting and animation. contour plot may need to be cleared after each frame is drawn. 
fig = vplot.Fig()
plt = fig[0,0]
img = plt.image(probSpace[0], cmap="coolwarm")
plt.colorbar(position="left", cmap=img.cmap, label="Probability", clim=[0,1])

#saveFile = cv2.VideoWriter("scatterAnim.mp4", cv2.VideoWriter_fourcc("F", "M", "P", "4"), 30, (1000, 1000))

"""command = ["ffmpeg",
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', "{}x{}".format(99, 99),
        '-pix_fmt', 'bgr32',
        '-r', "30",
        '-i', '-',
        '-an',
        '-vcodec', 'mpeg4',
        '-b:v', '5000k',
        "scatterSim.mp4"]"""

#proc = sp.Popen(command, stdin=sp.PIPE)

frame = 1
def update(ev):
    global frame
    global plt
    global img
    global probSpace

    if frame < len(t):
        img.clim="auto"
        img.set_data(probSpace[frame])
        img.update()

        frame += 1

timer = app.Timer(interval=0.02) 
timer.connect(update)
timer.start(interval=0.02)

#fig.show(run=True)
app.run()

#saveFile.release()

"""fig = plt.figure(figsize=(7, 7))


def animate(i):
	global x, y, probAmplitude
	cont = plt.contourf(x, y, probSpace[i], i)
	return cont

print("anim")
anim = animation.FuncAnimation(fig, animate, frames=len(probAmplitude), interval=20, blit=False, cache_frame_data=False)

'''
# saving the animation to a .mp4 file. 
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='UGQM'), bitrate=1800)
anim.save('im.mp4', writer=writer)
'''

plt.show()"""