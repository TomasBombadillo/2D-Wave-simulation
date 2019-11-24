# -*- coding: utf-8 -*-
"""
	Author: Tomás Londoño M.
	Description: 	simulatión de una onda en una membrana reflectiva y 
					un laser incidiendo verticalmente y siendo proyectado en una pantalla

	last edit: 18/11/19
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from wave_lib import *

#Variables globales
Nx = 200
Ny = 200
xinit = -3.36  #cm # -3.36
yinit = -3.36  #cm

dt = 0.0000001  # s
Nt = 40000
# ------------ Main ---------------------
t = 0
step = 100

tracker = 3e3

A = 0.1   # Amplitud of force
freq = [106, 155, 244, 312, 406] 	# frequency of force
v = 25e2    # wave velocity
center = np.random.random((2,1))

# Altura de pantalla
tubo = 8.7      #cm +-0.05
altura = 53.0   #cm +-0.1
Z = altura-tubo

# type of force
force = 'Bessel'	#'Uniform', 'Gaussian'

# Projection
#l = np.array([1.5, 1.5, 0])
l = np.array([[1.5, 1.5, 0.0],[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])

'''
Membrane = Wave(Nx, Ny, xinit, yinit, l, dt)

Membrane.init_wave(A, freq[0], v, center, force)

Membrane.init_projection(Z)

# Simulation

for t in range(0,Nt+1):
    Membrane.RK4(np.array([Membrane.phi,Membrane.pi,Membrane.psi_x,Membrane.psi_y]),t)
    
    if t%100==0: 
        print("{:0.6f}".format(t*Membrane.dt))
    
    if t%100==0:
        Membrane.projection()


plt.figure()
plt.title(r'Proyección en ({:0.2f}cm, {:0.2f}cm) en $t=${:0.4f}$s$ y $\nu=${:0.1f}$Hz$ '.format(Membrane.x_point,Membrane.y_point,t*Membrane.dt,Membrane.freq))
plt.axes(xlim=(l[0]-2,l[0]+2), ylim=(l[1]-2,l[0]+2))
plt.plot(Membrane.x_p,Membrane.y_p, '-')
plt.show()
'''
i=0
for p in l:
	i+=1
	for nu in freq:
		print("freq = {:0.1f} \n p=({:0.2f}, {:0.2f})\n ...Simulating...".format(nu,p[0],p[1]))
		center = np.random.random((2,))

		# init
		Membrane = Wave(Nx, Ny, xinit, yinit, p, dt)
		Membrane.init_wave(A, nu, v, center, force)
		Membrane.init_projection(Z)

		# Animations

		for t in range(0,Nt+1):
			Membrane.RK4(np.array([Membrane.phi,Membrane.pi,Membrane.psi_x,Membrane.psi_y]),t)

			if t%1000==0: 
			    print("{:0.6f}".format(t*Membrane.dt))

			if t%100==0:
			    Membrane.projection()

		plt.figure()
		plt.title(r'Proyección en ({:0.2f}cm, {:0.2f}cm) en $t=${:0.4f}$s$ y $\nu=${:0.1f}$Hz$ '.format(Membrane.x_point,Membrane.y_point,t*Membrane.dt,Membrane.freq))
		plt.xlim(p[0]-2,p[0]+2)
		plt.ylim(p[1]-2,p[1]+2)
		plt.plot(Membrane.x_p,Membrane.y_p, '-')
		plt.savefig('Sim_p{:0.0f}-{:0.0f}.png'.format(i,nu))
		

'''
# Animation of projection
fig = plt.figure()
plt.title('Projection in ({:0.2f}, {:0.2f})'.format(Membrane.x_point,Membrane.y_point))
ax = plt.axes(xlim=(l[0]-2,l[0]+2), ylim=(l[1]-2,l[0]+2))
line, = ax.plot([],[],'-')
ani_p = animation.FuncAnimation(fig, Membrane.animate_projection, fargs=(step,ax,line,int(tracker)), frames = range(t,t+Nt,step),blit=True)
plt.show()
'''
'''
fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111,projection='3d')
#ax = fig.gca(projection='3d')
ax = axes3d.Axes3D(fig)

ax.set_xlim3d([Membrane.Xinit,Membrane.Xinit+Membrane.Nx*Membrane.dx])
ax.set_xlabel('X')
ax.set_ylim3d([Membrane.Yinit,Membrane.Yinit+Membrane.Ny*Membrane.dy])
ax.set_ylabel('Y')
ax.set_zlim3d([-3,3])
ax.set_zlabel('Z')

ax.set_title('Wave con \nu={:0.1f}'.format(Membrane.freq))

#wave = ax.plot_wireframe(x,y,phi, rstride=2, cstride=2)
wave = ax.plot_surface(Membrane.x,Membrane.y,Membrane.phi, rstride=10, cstride=10)

#ani = animation.FuncAnimation(fig, make_animation, frames = range(Nt,Nt*2,1), fargs=(ax,fig),blit=False)
ani = animation.FuncAnimation(fig, Membrane.animate_wave, frames = range(t,t+Nt,step), fargs=(step,ax,wave),blit=False)
plt.show()
'''