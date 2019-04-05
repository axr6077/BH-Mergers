import numpy as np
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
import astropy.units as u
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde,rv_continuous
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord

r_lim = 40
r0 = 0.01

n_iter = 5000
n_cores = 16
int_time = 10
orbit_pnts = 10000

rho_0b = 1.0719
rho_0d = 2.6387
rho_0s = 13.0976

q = 0.6
gamma = 1.8
rt = 1.9
rm = 6.5
rd = 3.5
rz = 0.41
bs = 7.669
Re = 2.8

name = 'SwiftJ1753'
w = -0.01
s = 0.13
RA = 268.367874
DEC = -1.451738
pm_ra = 0.9
pm_ra_err = 0.06
pm_dec = -3.64
pm_dec_err = 0.06
V_rad = 130
V_rad_err = 10

c_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')
l = c_icrs.galactic.l.deg
b = c_icrs.galactic.b.deg


def pass_v(ra, dec, d, pm_ra, pm_dec, V, time, numpoints):
    o = Orbit([ra * u.deg, dec * u.deg, d * u.kpc, pm_ra * u.mas / u.yr, pm_dec * u.mas / u.yr, V * u.km / u.s],
              radec=True)
    lp = MWPotential2014
    ts = np.linspace(0, time, numpoints) * u.Gyr
    o.integrate(ts, lp)

    pass_t_array = ts[np.where(np.sign(o.z(ts)[:-1]) - np.sign(o.z(ts)[1:]) != 0)[0]]
    results = []
    for pass_t in pass_t_array:
        o2 = Orbit(vxvv=[o.R(pass_t) / 8.0, 0., 1., 0., 0., o.phi(pass_t)], ro=8., vo=220.)
        # results.append(np.sqrt((o.U(pass_t)-o2.U(0)+11.1)**2 + (o.V(pass_t)-o2.V(0)+12.24)**2 + (o.W(pass_t)-o2.W(0)+7.25)**2))
        results.append(
            np.sqrt((o.U(pass_t) - o2.U(0)) ** 2 + (o.V(pass_t) - o2.V(0)) ** 2 + (o.W(pass_t) - o2.W(0)) ** 2))

    return results


def peculiar(ra, dec, d, pm_ra, pm_dec, V):
    o = Orbit([ra * u.deg, dec * u.deg, d * u.kpc, pm_ra * u.mas / u.yr, pm_dec * u.mas / u.yr, V * u.km / u.s],
              radec=True)
    o2 = Orbit(vxvv=[o.R(0.) / 8.0, 0., 1., 0., 0., o.phi(0.)], ro=8., vo=220.)
    current_vel = np.sqrt(
        (o.U(0.) - o2.U(0) + 11.1) ** 2 + (o.V(0.) - o2.V(0) + 12.24) ** 2 + (o.W(0.) - o2.W(0) + 7.25) ** 2)
    return current_vel

def bulge(r,z):
    k = (r**2 + ((z**2)/(q**2)))
    return rho_0b*((np.sqrt(k))**(-gamma))*np.exp(-k/(rt**2))
def disk(r,z):
    return rho_0d*(np.exp(-(rm/rd)-(r/rd)-(np.abs(z)/rz)))
def sphere(R):
    return rho_0s*(np.exp(-bs*((R/Re)**(1.0/4)))/((R/Re)**(7.0/8.0)))

def new_prior(x,l,b,r_lim):
    if x > 0 and x <= r_lim:
        z = x*np.sin(np.radians(b))
        R0 = 8
        r = np.sqrt(R0**2 + ((x*np.cos(np.radians(b)))**2) - 2*x*R0*np.cos(np.radians(b))*np.cos(np.radians(l)))
        R = np.sqrt(R0**2 + (x**2) - 2*x*R0*np.cos(np.radians(b))*np.cos(np.radians(l)))
        rho_b = bulge(r,z)
        rho_d = disk(r,z)
        rho_s = sphere(R)
        return (rho_b+rho_d+rho_s)*((x*1e3)**2)
    else:
        return 0

def posterior(x,w,s,r_lim):
    return new_prior(x,l,b,r_lim)*likelihood(x,w,s)

def likelihood(x,w,s):
    return 1 / (np.sqrt(2 * np.pi) * s) * np.exp(-1 / (2 * s ** 2) * (w - 1 / x) ** 2)

def normalization(f,par1,w,s,p,r_mode,par2):
    N = quad(f, r_mode, par2, args=(w, s, par1), epsrel=1e-11,
             epsabs=1e-11)
    N = N[0] + p

    return N

def percentiles(f,r0,r_mode,w,s,par):
    p = quad(f, r0, r_mode, args=(w, s, par), epsrel=1e-12, epsabs=1e-12)
    return p[0]


x = np.arange(r0,r_lim,r_lim/n_iter)
y1 = [posterior(i,w,s,r_lim) for i in x]
r_mode=x[y1.index(max(y1))]
plt.plot(x,y1)
print(r_mode)
plt.savefig(str(name)+'posterior.png')

p = percentiles(posterior, r0, r_mode, w, s,r_lim)  # Computing the percentile that corresponds to the mode of the PDF
n = normalization(posterior, r_lim, w, s, p, r_mode,r_lim)  # Computing the normalization constant of the PDF
x = np.arange(r0,r_lim,r_lim/n_iter)
y2 = [posterior(i,w,s,r_lim)/n for i in x]
testf = interp1d(x,y2)

class custom_dist_analytical(rv_continuous):
    def _pdf(self, xx):
        return testf(xx)

np.random.seed(124)

src_ra = np.random.normal(RA,0, n_iter)
src_dec = np.random.normal(DEC,0,n_iter)
src_d = custom_dist_analytical(a=r0,b=r_lim).rvs(size=n_iter)
src_pm_ra = np.random.normal(pm_ra,pm_ra_err,n_iter)
src_pm_dec = np.random.normal(pm_dec,pm_dec_err,n_iter)
src_V = np.random.normal(V_rad,V_rad_err,n_iter)
v_dist = Parallel(n_jobs=n_cores,verbose=5)(delayed(pass_v)(src_ra[i],src_dec[i],src_d[i],src_pm_ra[i],src_pm_dec[i],src_V[i],int_time,orbit_pnts) for i in range(n_iter))
total_dist = np.concatenate(v_dist[:])

v_c = Parallel(n_jobs=n_cores,verbose=5)(delayed(peculiar)(src_ra[i],src_dec[i],src_d[i],src_pm_ra[i],src_pm_dec[i],src_V[i]) for i in range(n_iter))

np.savetxt(str(name)+'_natalkicks_dist.txt',np.transpose(total_dist),delimiter=' ')
np.savetxt(str(name)+'_distance_dist.txt',np.transpose(src_d),delimiter=' ')

median = np.percentile(total_dist,50)
low_lim = np.percentile(total_dist,5)
high_lim = np.percentile(total_dist,95)
print(median,low_lim,high_lim)

plt.title(str(name))
histplot = plt.hist(total_dist,bins=50,color='orchid',edgecolor='w',density=True,alpha=0.7)
plt.vlines([low_lim,high_lim],0,max(histplot[0])*1.1,linestyles=':',colors='midnightblue')
plt.vlines(median,0,max(histplot[0])*1.1,linestyles='--',colors='midnightblue')
plt.ylim(0,max(histplot[0])*1.1)
plt.ylabel('Probability density', fontsize=12)
plt.xlabel('Potential natal kick (km s$^{-1}$)', fontsize=12)
plt.minorticks_on()
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='major', length=9)
plt.tick_params(axis='both', which='minor', length=4.5)
plt.tick_params(axis='both', which='both',direction='in',right=True,top=True)
plt.savefig(str(name)+'_natalkick_dist.png')

median = np.percentile(src_d,50)
low_lim = np.percentile(src_d,5)
high_lim = np.percentile(src_d,95)
print(median,low_lim,high_lim)
histplot = plt.hist(src_d,bins=60,color='teal',edgecolor='w',density=True,alpha=0.7)
plt.vlines(median,0,max(histplot[0])*1.1,linestyles=':',colors='midnightblue')
plt.vlines(r_mode,0,max(histplot[0])*1.1,linestyles='--',colors='black')
plt.ylim(0,max(histplot[0])*1.1)
plt.ylabel('Probability density', fontsize=12)
plt.xlabel('Distance (kpc)', fontsize=12)
plt.title(str(name))
plt.minorticks_on()
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='major', length=9)
plt.tick_params(axis='both', which='minor', length=4.5)
plt.tick_params(axis='both', which='both',direction='in',right=True,top=True)
plt.savefig(str(name)+'_distance_dist.png')

histplot = plt.hist(src_V,bins=50,color='orange',edgecolor='w',density=True,alpha=0.7)
plt.ylim(0,max(histplot[0])*1.1)
plt.ylabel('Probability density', fontsize=12)
plt.xlabel('Radial velocity (km/s)', fontsize=12)
plt.title(str(name))
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='major', length=9)
plt.tick_params(axis='both', which='minor', length=4.5)
plt.tick_params(axis='both', which='both',direction='in',right=True,top=True)
plt.savefig(str(name)+'_Vrad_dist.png')


plt.plot(x,testf(x))
plt.ylabel('Probability density', fontsize=12)
plt.xlabel('Distance (kpc)', fontsize=12)
plt.savefig(str(name)+'_posterior_dist.png')

median = np.percentile(v_c,50)
low_lim = np.percentile(v_c,5)
high_lim = np.percentile(v_c,95)
print(median,low_lim,high_lim)
histplot = plt.hist(v_c,bins=180,color='orange',edgecolor='w',density=True,alpha=0.7)
plt.vlines([low_lim,high_lim],0,max(histplot[0])*1.1,linestyles=':',colors='midnightblue')
plt.vlines(median,0,max(histplot[0])*1.1,linestyles=':',colors='midnightblue')
plt.ylim(0,max(histplot[0])*1.1)
plt.ylabel('Probability density', fontsize=12)
plt.xlabel('Peculiar velocity', fontsize=12)
plt.xlim(0,400)
plt.title(str(name))
plt.minorticks_on()
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='major', length=9)
plt.tick_params(axis='both', which='minor', length=4.5)
plt.tick_params(axis='both', which='both',direction='in',right=True,top=True)
plt.savefig(str(name)+'_peculiar_velocity.png')

prob = histplot[0]
bins = np.array(histplot[1])
prob==np.max(prob)
mode=bins[np.where(prob ==max(prob))]
print(mode)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X (kpc)')
ax.set_ylabel('Y (kpc)')
ax.set_zlabel('Z (kpc)');
#ax.set_xlim(-10,10)
#ax.set_ylim(-10,10)
#ax.set_zlim(-10,10)
for i in range(20):
    samp_o = Orbit([src_ra[i]*u.deg,src_dec[i]*u.deg,src_d[i]*u.kpc,src_pm_ra[i]*u.mas/u.yr,src_pm_dec[i]*u.mas/u.yr,src_V[i]*u.km/u.s],radec=True)
    lp= MWPotential2014
    ts= np.linspace(0,1.0,10000)*u.Gyr
    samp_o.integrate(ts,lp)
    if src_d[i]<10:
        ax.plot(samp_o.x(ts),samp_o.y(ts),samp_o.z(ts),c='b',linewidth=0.2,alpha=0.5)
plt.savefig(str(name)+'3d_orbits.png')

z = []
r = []
for i in range (0,len(src_d)):
    orbits_z = Orbit([src_ra[i]*u.deg,src_dec[i]*u.deg,src_d[i]*u.kpc,src_pm_ra[i]*u.mas/u.yr,src_pm_dec[i]*u.mas/u.yr,src_V[i]*u.km/u.s],radec=True)
    lp= MWPotential2014
    ts= np.linspace(0,1.0,500)*u.Gyr
    orbits_z.integrate(ts,lp)
    z.append(orbits_z.zmax())
    r.append(src_d[i])
np.savetxt(str(name)+'_zmax_dist.txt',np.transpose(z),delimiter=' ')
plt.plot(r,z)
plt.ylabel('Z max during an orbit', fontsize=12)
plt.xlabel('Distance (kpc)', fontsize=12)
plt.savefig(str(name)+'zmax_and_dist.png')

median = np.percentile(z,50)
low_lim = np.percentile(z,5)
high_lim = np.percentile(z,95)
print(median,low_lim,high_lim)
plt.hist(z,bins=200,color='orange',edgecolor='w',density=True,alpha=0.7)
plt.xlim(0,100)

z = []
r = []
ts= np.linspace(0,1.0,500)*u.Gyr
lp= MWPotential2014
for i in range (0,len(src_d)):
    orbits_z = Orbit([src_ra[i]*u.deg,src_dec[i]*u.deg,src_d[i]*u.kpc,src_pm_ra[i]*u.mas/u.yr,src_pm_dec[i]*u.mas/u.yr,src_V[i]*u.km/u.s],radec=True)
    z.append(orbits_z.z(0.))
    r.append(src_d[i])
np.savetxt(str(name)+'_z_dist.txt',np.transpose(z),delimiter=' ')

median = np.percentile(z,50)
low_lim = np.percentile(z,5)
high_lim = np.percentile(z,95)
print(median,low_lim,high_lim)
plt.hist(z,bins=100,color='orange',edgecolor='w',density=True,alpha=0.7)
print(np.std(z, ddof=1))
plt.savefig(str(name)+'_z_dist.png')

#EOF