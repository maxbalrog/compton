import numpy as np
import matplotlib.pyplot as plt
import scipy

from scipy.special import jv
from scipy.interpolate import interp1d
from scipy.fftpack import fft
from numpy.fft import fftfreq
from scipy import integrate

import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


envelope_type = 'gauss'

'''
This cell exploits analytical calculation of trajectories (u and x) of electron
motion in laboratory frame
'''

#@jit(nopython=True)
def g(eta, tau, fl=envelope_type):
    if fl == 'gauss':
        res = np.exp(-eta**2/tau**2)
    elif fl == 'super gauss':
        res = np.exp(-eta**60/tau**60)
    elif fl == 'sin':
        n = np.shape(eta)[0]
        res = np.zeros(n)
        ind = np.where((eta > 0) & (eta < tau))
        res[ind] = (np.sin(np.pi * eta[ind] / tau))**2

    return res

#@njit
def A_(eta, a0, delta, tau):
    n = np.shape(eta)[0]
    A = np.zeros((n,3))
    A[:,0] = a0 * g(eta, tau) * np.cos(eta)
    A[:,1] = a0 * g(eta, tau) * delta * np.sin(eta)

    return A

#@njit
def u_(eta, a0, delta, tau, u0, gamma0):
    n = np.shape(eta)[0]
    u = np.zeros((n,3))
    A = A_(eta, a0, delta, tau)

    pi0 = gamma0 - u0[2]

    u[:,0] = u0[0] + A[:,0]
    u[:,1] = u0[1] + A[:,1]
    u[:,2] = u0[2] + (u0[0] * A[:,0] + u0[1] * A[:,1] + 0.5 * (A[:,0]**2 + A[:,1]**2)) / pi0

    return u

#@jit
def create_dynamics(eta, a0, delta, tau, d_eta, u0=np.zeros(3), gamma0=1, x0=np.zeros(3)):
    n = np.shape(eta)[0]
    u_r = np.zeros((n,7))
    u_r[:,0] = eta.copy()

    u_r[:,1:4] = u_(eta, a0, delta, tau, u0, gamma0)

    pi0 = gamma0 - u0[2]

    u_r[:,4] = x0[0] + integrate.cumtrapz(u_r[:,1]/pi0, dx=d_eta, initial=0)
    u_r[:,5] = x0[1] + integrate.cumtrapz(u_r[:,2]/pi0, dx=d_eta, initial=0)
    u_r[:,6] = x0[2] + integrate.cumtrapz(u_r[:,3]/pi0, dx=d_eta, initial=0)

    return u_r

def ft(samples, Fs, t0):
    """Approximate the Fourier Transform of a time-limited
    signal by means of the discrete Fourier Transform.

    samples: signal values sampled at the positions t0 + n/Fs
    Fs: Sampling frequency of the signal
    t0: starting time of the sampling of the signal
    """
    f = np.linspace(-Fs/2, Fs/2, len(samples), endpoint=False)
    return np.fft.fftshift(np.fft.fft(samples)/Fs * np.exp(2j*np.pi*f*t0))

def spec_I_w_many(eta, a0, delta, tau, d_eta, t0, n_t_points, n_t_padded, u0,
                  gamma0, x0, rank, theta=np.pi, fi=0, t_end=0):

    n_electrons = u0.shape[0]

    #calculating trajectories
    u_r = create_dynamics(eta, a0, delta, tau, d_eta,
                          u0=u0[rank,:], gamma0=gamma0[rank,0], x0=x0[rank,:])

    pi0 = gamma0[rank,0] - u0[rank,2]

    u_x_points = u_r[:,1]
    u_y_points = u_r[:,2]
    u_z_points = u_r[:,3]

    x_points = u_r[:,4]
    y_points = u_r[:,5]
    z_points = u_r[:,6]

    #new time corresponding to \eta and construction of adaptive grid in t
    t_eta_points = eta + (1 - np.cos(theta))*z_points - x_points * np.cos(fi) * np.sin(theta) - y_points * np.sin(fi) * np.sin(theta)

    #if t_end == 0:f = f[f>=0]
    t_end = max(t_eta_points)
    t = np.linspace(-t0, t_end, n_t_points)

    #interpolation of \eta(t)
    eta_interp = interp1d(t_eta_points, eta, kind='cubic')

    #interpolation of trajectories
    u_x = interp1d(eta, u_x_points, kind='cubic')
    u_y = interp1d(eta, u_y_points, kind='cubic')
    u_z = interp1d(eta, u_z_points, kind='cubic')

    x = interp1d(eta, x_points, kind='cubic')
    y = interp1d(eta, y_points, kind='cubic')
    z = interp1d(eta, z_points, kind='cubic')

    #integrand in retarded time
    u_x_ret = u_x(eta_interp(t))
    u_y_ret = u_y(eta_interp(t))
    u_z_ret = u_z(eta_interp(t))

    #Jacobian of time transform in retarded time
    Jacobian_ret = 1 + ((1 - np.cos(theta))*u_z_ret - u_x_ret*np.cos(fi)*np.sin(theta) - u_y_ret*np.sin(fi)*np.sin(theta)) / pi0

    #Samples of Ix integrals on t grid
    Ix_samples = u_x_ret / Jacobian_ret / pi0
    Iy_samples = u_y_ret / Jacobian_ret / pi0
    Iz_samples = u_z_ret / Jacobian_ret / pi0

    #padding with zeros
    Ix_samples = np.pad(Ix_samples, n_t_padded, 'constant')
    Iy_samples = np.pad(Iy_samples, n_t_padded, 'constant')
    Iz_samples = np.pad(Iz_samples, n_t_padded, 'constant')

    #sampling frequency and frequency range
    Fs = n_t_points / (t0+t_end)
    f = np.linspace(-Fs/2, Fs/2, len(Ix_samples), endpoint=False)

    #new start time of sampling
    t0_new = t0 + n_t_padded/Fs

    #fft for positive frequncies
    Ix = ft(Ix_samples, Fs, t0_new)[f>=0]
    Iy = ft(Iy_samples, Fs, t0_new)[f>=0]
    Iz = ft(Iz_samples, Fs, t0_new)[f>=0]

    w = 2 * np.pi * f[f>=0]

    Iw = np.zeros((4,len(Ix)), dtype=complex)
    Iw[0] = w
    Iw[1] = Ix
    Iw[2] = Iy
    Iw[3] = Iz

    return Iw


a0 = 1
delta = 0

theta = np.pi
fi = 0

#electrons' initial parameters
n_electrons = 4

#v0 = np.array([[.0,.0,-0.80]])
v0 = np.array([[.0,.0,-0.80],
               [.0,.0,-0.82],
               [.0,.0,-0.85],
               [.0,.0,-0.78]])
gamma0 = 1. / np.sqrt(1. - np.linalg.norm(v0, axis=1))
gamma0 = gamma0.reshape((n_electrons,1))
u0 = v0 * gamma0

#x0 = np.array([[.0,.0,.0]])
x0 = np.array([[.0,.0,.0],
               [.0,.0,.0],
               [.0,.0,.0],
               [.0,.0,.0]])

#pulse parameters
tau = 5 * 2 * np.pi      #length of a pulse
eta_end = 150             #boundary of simulation grid in \eta
n_eta = 2*eta_end*2000     #number of points in \eta grid
d_eta = 2*eta_end/n_eta
eta = np.linspace(-eta_end, eta_end, n_eta)  #\eta grid

#parameters for grid in self-time
t0 = eta_end
n_t_points = t0*100         #number of points in t grid
n_t_padded = 20*n_t_points  #padded length

if rank == 0:
    start = time.time()

Iw = spec_I_w_many(eta, a0, delta, tau, d_eta, t0, n_t_points, n_t_padded, u0, gamma0, x0, rank)

if rank == 0:
    end = time.time() - start
    f = open('timings.txt', 'a+')
    f.write('%d %.6f\n' %(n_electrons,end))
    f.close()

np.savetxt('Iw_%d' %rank, Iw)
