from os.path import join
import netCDF4
import numpy as np
from scipy.stats import circmean
import matplotlib.pyplot as plt
path = join("data","AirMSPI_ER2_GRP_ELLIPSOID_20130206_203133Z_Pacific-32N123W_660A_F01_V003.hdf")
f = netCDF4.Dataset(path, diskless=True, persist=False)

channels_data = f['HDFEOS']['GRIDS']
image = np.array(channels_data['660nm_band']['Data Fields']['I'])

sun_azimuth = np.array(channels_data['660nm_band']['Data Fields']['Sun_azimuth'])
sun_zenith = np.array(channels_data['660nm_band']['Data Fields']['Sun_zenith'])


sun_azimuth = circmean(sun_azimuth[image > 0].ravel(), high=360)
sun_zenith_list = 180 - circmean(sun_zenith[image > 0].ravel(), high=360)

channels_data_ancillary = f['HDFEOS']['GRIDS']['Ancillary']['Data Fields']
latitude = np.array(channels_data_ancillary['Latitude'])
longitude = np.array(channels_data_ancillary['Longitude'])
if 'Elevation' in f['HDFEOS']['GRIDS']['Ancillary']['Data Fields'].variables.keys():
    ground_height = np.array(channels_data_ancillary['Elevation'])  # [m]
else:
    ground_height = np.full(longitude.shape, 0)

zenith = np.array(channels_data['660nm_band']['Data Fields']['View_zenith'])
azimuth = np.array(channels_data['660nm_band']['Data Fields']['View_azimuth'])

image[image == -999] = 0
title = path.split('/')[-1].split('.')[0].split('ELLIPSOID_')[-1]
plt.imshow(image, cmap="gray")
plt.show()