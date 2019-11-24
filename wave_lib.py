# -*- coding: utf-8 -*-
"""
  Author: Tomás Londoño M.
  Description:  Librería para la simulación de onda

  last edit: 18/11/19
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from scipy.special import jv

class Wave:
  # Clase que contiene a la onda juntos con sus propiedades
  # y sus funciones

  # error
  eps = 1e-15

  def __init__(self,Nx,Ny,Xinit,Yinit,pos,dt):
    def closest(lst, K): 
      lst = np.asarray(lst) 
      idx = (np.abs(lst - K)).argmin() 
      return idx
    self.Nx = Nx
    self.Ny = Ny
    self.Xinit = Xinit  # cm
    self.Yinit = Yinit  # cm

    self.dx = -Xinit*2/Nx   # cm
    self.dy = -Yinit*2/Ny   # cm

    self.dt = dt  # s

    xtemp = np.arange(self.Xinit,-self.Xinit,self.dx)
    ytemp = np.arange(self.Yinit,-self.Yinit,self.dy)

    # plane
    self.x, self.y = np.meshgrid(xtemp,ytemp)   # cm

    self.x2 = closest(ytemp,pos[0])
    self.x1 = closest(xtemp,pos[1])

  def init_wave(self, A, freq, v, center, method):
    self.A = A
    self.freq = freq  # Hz
    #self.v = A*freq # cm/s
    self.v = v   #cm/s

    self.center = center

    self.f_method = method

    # Circle matrix
    rlength = self.Xinit+self.Nx*self.dx
    self.circle_grid = draw_noncentered_circle(int(self.Nx/2),self.Nx/rlength, 2)

    # circle matrix of force multiplier
    if method=='Gaussian':
      self.force_circle = np.exp((-(self.x-self.center[0])**2.-(self.y-self.center[1])**2.)/0.1)*draw_noncentered_circle(int(self.Nx/2),self.Nx/rlength, 4)
    elif method=='Uniform':
      self.force_circle = draw_noncentered_circle(int(self.Nx/2),self.Nx/rlength, 4)
    elif method=='Bessel':
      self.force_circle = jv(0, (-(self.x-self.center[0])**2.-(self.y-self.center[1])**2.)/0.5 )*draw_noncentered_circle(int(self.Nx/2),self.Nx/rlength, 4)
    

    # Initial conditions
    #phi1 = 0.1*np.exp((-(self.x-self.center[0])*(self.x-self.center[0])-(self.y-self.center[1])*(self.y-self.center[1]))/3)*self.circle_grid + self.eps # cm
    phi1 = 0.1*jv(0, (-(self.x-self.center[0])*(self.x-self.center[0])-(self.y-self.center[1])*(self.y-self.center[1]))/0.1 )*self.circle_grid
    phi0 = np.zeros_like(self.x) + self.eps # cm
    #phi0 = self.A*0.5*np.exp((-self.x*self.x-self.y*self.y)/3)*self.circle_grid + self.eps # cm
    self.phi = phi0

    # variable change
    self.psi_x = np.gradient(self.phi,self.dx,axis=1)
    self.psi_y = np.gradient(self.phi,self.dy,axis=0)
    self.pi    = np.zeros_like(self.phi) # cm/s
    self.pi = (phi0-phi1)/(2000*self.dt) + self.eps*np.ones_like(self.phi)

  def init_projection(self, height):
    self.Z = height

    self.x_p = []
    self.y_p = []

    self.x_point = self.x[self.x1,self.x2]
    self.y_point = self.y[self.x1,self.x2]
    self.z_point = self.phi[self.x1,self.x2]
    self.zp_dx = self.psi_x[self.x1,self.x2]
    self.zp_dy = self.psi_x[self.x1,self.x2]

  def projection(self):
    self.z_point = self.phi[self.x1,self.x2]
    self.zp_dx = self.psi_x[self.x1,self.x2]
    self.zp_dy = self.psi_x[self.x1,self.x2]
    L = np.array([0,0,1]) # vertical beam
    N = np.array([-self.zp_dx,-self.zp_dy,1])

    #angle between laser vector and normal vector
    theta = np.arccos(np.dot(N,L) / (np.sqrt(np.dot(N,N))*np.sqrt(np.dot(L,L))))  # radians 
    
    # normal axis to rotation plane
    u_x,u_y,u_z = np.cross(L,N) 

    # calculations
    cos = np.cos(2*-theta)
    sin = np.sin(2*-theta)
    t = 1.-cos

    # Rotation matrix for arbitrary axis
    R1 = np.array([cos+u_x**2. * t, u_x*u_y*t-u_z*sin, u_x*u_z*t+u_y*sin])
    R2 = np.array([u_x*u_y*t+u_z*sin, cos+u_y**2. *t, u_y*u_z*t-u_x*sin])
    R3 = np.array([u_x*u_z*t-u_y*sin, u_y*u_z*t+u_x*sin, cos+u_z**2. *t])

    R = np.array([R1,R2,R3])

    L_r = np.dot(L.T, R)

    # hallar coordenadas de la proyección x_p y y_p
    # vector de direccióón d = (l,m,n) y pasa por el punto (x1,y1,z1)
    # usando (x - x1)/l = (y - y1)/m = (z - z1)/n
    x1, y1, z1 = self.x_point,self.y_point,self.z_point
    l, m, n = L_r[0], L_r[1], L_r[2]

    self.x_p.append(l*(self.Z - z1)/n + x1)
    self.y_p.append(m*(self.Z - z1)/n + y1)

    #print("{:0.6f},{:0.6f}".format(self.x_p[-1],self.y_p[-1]))

    #return N, L_r, x_p, y_p

  def RK4(self,y0,t):
    t0 = t*self.dt
    k0 = self.Derivate(t0,y0)
    k1 = self.Derivate(t0+self.dt/2.,y0+k0*self.dt/2.)
    k2 = self.Derivate(t0+self.dt/2.,y0+k1*self.dt/2.)
    k3 = self.Derivate(t0+self.dt,y0+k2*self.dt)

    y1 = y0+(self.dt/6.)*(k0+2.*k1+2.*k2+k3)

    self.phi = y1[0]
    self.pi = y1[1]
    self.psi_x = y1[2]
    self.psi_y = y1[3]

  def Derivate(self,t,x): # Derivatives function
    phi = x[0]
    pi  = x[1]
    psix = x[2]
    psiy = x[3]

    phi = np.multiply(self.circle_grid,phi)
    pi = np.multiply(self.circle_grid,pi)
    psix = np.multiply(self.circle_grid,psix)
    psiy = np.multiply(self.circle_grid,psiy)

    sphi = pi  

    spi  = self.v**2.*(np.gradient(psix,self.dx,axis=1) + np.gradient(psiy,self.dy,axis=0)) + self.Force(t,self.f_method)
    spsix = np.gradient(pi,self.dx,axis=1) #(pi[2:]-pi[:-2])/(2.*dx)
    spsiy = np.gradient(pi,self.dy,axis=0)

    return np.array([sphi,spi,spsix,spsiy])

  def Force(self, t, method):
      omega = self.freq*2.*np.pi
      if method=='Gaussian':
        f =  self.A* np.sin(omega*t)*self.force_circle
      elif method=='Uniform':
        f =  self.A* np.sin(omega*t)*self.force_circle
      elif method=='Bessel':
        f = self.A* np.sin(omega*t)*self.force_circle 
      
      return f

  def animate_projection(self, t,step,ax,draw,tracker):
    def animate(t,ax,draw,track):
      #self.RK4(np.array([self.phi,self.pi,self.psi_x,self.psi_y]),t)

      self.projection()

      if len(self.x_p)<track:
          draw.set_data(self.x_p,self.y_p)
      else:
          draw.set_data(self.x_p[t-track:],self.y_p[t-track:])
      #print("{:0.4f} -----> ({:0.6f}, {:0.6f})".format(t*self.dt,self.x_p[-1], self.y_p[-1]))
      return draw,

    for i in range(t,t+step,1):
      self.RK4(np.array([self.phi,self.pi,self.psi_x,self.psi_y]),i)
      #print("{:0.7f}".format(i*self.dt))
    draw, = animate(i,ax,draw,tracker)
    return draw,

  def animate_wave(self,t,step,ax,draw):
    def animate(t,ax,draw):
      ax.clear()
      ax.set_xlim3d([self.Xinit,self.Xinit+self.Nx*self.dx])
      ax.set_xlabel('X')
      ax.set_ylim3d([self.Xinit,self.Xinit+self.Ny*self.dy])
      ax.set_ylabel('Y')
      ax.set_zlim3d([-3,3])
      ax.set_zlabel('Z')

      ax.set_title('Wave in {:0.6}s'.format(t*self.dt))

      #self.RK4(np.array([self.phi,self.pi,self.psi_x,self.psi_y]),t)

      #self = ax.plot_wireframe(x,y,phi, rstride=2, cstride=2)
      draw = ax.plot_surface(self.x,self.y,self.phi, rstride=20, cstride=20)

      return draw,
    for i in range(t,t+step,1):
      self.RK4(np.array([self.phi,self.pi,self.psi_x,self.psi_y]),i)
      #print("{:0.7f}".format(i*self.dt))

    draw, = animate(i,ax,draw)
    #print("{:0.7f}".format(t*self.dt))
    
    return draw,

""" Draws noncentered circle in cell """
def draw_noncentered_circle(r_p, ratio,less):
    # r_p: Radius in pixels
    # ratio: Pixels/length total ratio
    
    # x_p: abs(x_p) < r_p Shift in x coordinate
    # y_p: abs(y_p) < r_p Shift in y coordinate
    
    x, y = np.indices((2*r_p, 2*r_p))
    
    delta = 1.17e-5 #Maximum deviation from center
    delta_p = delta*ratio #Maximum deviation in pixels
    
    np.random.seed()
    
    #Deviation from center (random)
    x_p = 2*delta_p*np.random.random() - delta_p
    y_p = 2*delta_p*np.random.random() - delta_p
    
    #circle = (x - 4*r_p/2 - x_p)**2 + (y - 4*r_p/2 - y_p)**2 < (r_p)**2
    circle = (x - r_p - x_p)**2 + (y - r_p - y_p)**2 < (r_p-less)**2
    circle = circle.astype(float)
    
    return circle
    # circle: Matrix. Cell with noncentered circle
