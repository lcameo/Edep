#http://nptfit.readthedocs.io/en/latest/Example4_Simple_NPTF.html

#%matplotlib inline
#%load_ext autoreload
#%autoreload 2


from __future__ import print_function

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import rcParams

from NPTFit import nptfit # module for performing scan
from NPTFit import dnds_analysis # module for analysing the output
from matplotlib import pyplot as plt # to see plots

import numpy as np

#Example 1: A map without point sources

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def truncated_power_law(a, m):
    E = np.arange(1, m+1,dtype='float') #array of energy values
    pmf = 1/E**a                        #math with arrays like numbers, thanks python!
    pmf /= pmf.sum()                    #normalizes the distribution
    return stats.rv_discrete(values=(range(1, m+1), pmf)) #80/20.. this goes in the 20. keep er movin. 

def get_spectral_coefficients(a,m):
	d = truncated_power_law(a,m)#output 
	
	N = 10**4
	sample = d.rvs(size=N)  #is an array of the randomly generated values

	bins = np.arange(m)+0.5   #there will be as many bins as energy values ....+0.5 keep er movin
	hist1, _, _ = plt.hist(sample, bins, normed=True)
	plt.title('Photon Energy Distribution')

	first_spectral_coeff = sum(hist1[0:(len(bins)-1)/2])
	second_spectral_coeff = sum(hist1[(len(bins)-1)/2:len(bins)-1])

	return first_spectral_coeff, second_spectral_coeff


#mean is an array of the mean number of photons counts in pixel p, it is the usual poissonian parameter
#spectrals an array of spectral coefficients (lambda_1, lambda_2 . . .) as many as you have energy bins
#energy data is the number of photons in each pixel in each bin of energy, it is a 2D array
#[pixels,energy bins] so energy_data[3,1] is the number of counts in the 4th pixel and 2nd energy bin
def loglike(mean, spectrals, energy_data):


	data = np.sum(energy_data, axis=0)
	ll = 0

	for p in range(0,len(mean)):
		#print('pixel ', p)
		ll += -mean[p] + data[p]*np.log(mean[p]) #- lgamma(data[p] + 1.) comes from cython and cython doesn't work
		for s in range(0,len(spectrals)):
			#print('energy bin ', s)
			ll += energy_data[s][p]

	return ll

a, m = .5, 3 
first_spectral_coeff, second_spectral_coeff = get_spectral_coefficients(a,m)
print('first spectral coefficient = ', first_spectral_coeff)
print('second spectral coefficient = ', second_spectral_coeff)

nside = 2
npix = hp.nside2npix(nside)
data = np.zeros(npix)

print('number of pixels = ', npix)

#Uncomment below to make poissonian data
data = np.random.poisson(1,npix)

#Uncomment below to make non poissonian point source data
#for ips in range(10):
#	data[np.random.randint(npix)] += np.random.poisson(50)

#Split the data up into energy bins	
data_bin1 = data*first_spectral_coeff  #This is the number of photons in the first energy bin according to the power law
								   #Note: Fractional number of photons, maybe non physical?
data_bin2 = data*second_spectral_coeff #This is the number of photons in the second energy bin according to the power law
exposure = np.ones(npix)

max_counts = np.amax(data)

hp.mollview(data_bin1, title='Fake Data, photons in energy bin 1', min=0, max=max_counts)
hp.mollview(data_bin2, title='Fake Data, photons in energy bin 2', min=0, max=max_counts)
hp.mollview(data, title='Fake Data, total number of photons', min=0, max=max_counts)
hp.mollview(data_bin1 + data_bin2, 'Fake Data, photons in energy bin 1 + photons in energy bin 2', min=0, max=max_counts)
#hp.mollview(exposure,title='Exposure Map')


spectrals = [0.5973, 0.40269]
mean = np.ones(npix)
energy_data = [data_bin1, data_bin2]

print('the log likehood = ', loglike(mean, spectrals, energy_data))

plt.plot()
plt.show()

print('bin1 = ', data_bin1)
print('bin2 = ', data_bin2)

iso = np.ones(npix)

n = nptfit.NPTF(tag='SimpleExample')
n.load_data(data,exposure)
n.add_template(iso, 'iso_p', units='flux')
#n.add_template(iso, 'iso_np', units='PS')

n.add_poiss_model('iso_p', '$A_\mathrm{iso}$', [0,2], False)
#n.add_non_poiss_model('iso_np', ['$A^\mathrm{ps}_\mathrm{iso}$', '$n_1$', '$n_2$', '$S_b$'], [[-10,1], [2.05, 60], [-60,1.95], [0.01,200]], [True, False, False, False])


n.configure_for_scan()
n.perform_scan(nlive=500)

n.load_scan()
an = dnds_analysis.Analysis(n)
an.make_triangle()
plt.show()
plt.close()

an.plot_intensity_fraction_poiss('iso_p', bins=20, color='cornflowerblue', label='Poissonian')
an.plot_intensity_fraction_non_poiss('iso_np', bins=20, color='firebrick', label='non-Poissonian')
plt.xlabel('Flux fraction (\%)')
plt.legend(fancybox = True)
plt.xlim(0,100);
plt.ylim(0,0.4);


