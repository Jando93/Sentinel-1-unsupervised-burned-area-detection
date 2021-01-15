#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('C:\\.snap\\snap-python\\snappy')
import snappy


# In[2]:


from os.path import join
import subprocess
from glob import iglob
from zipfile import ZipFile
import numpy as np
import pandas as pd
import snappy
from snappy import jpy, GPF, ProductIO
import matplotlib.pyplot as plt


# In[3]:


#set target folder and extract metadata
#product_path = '...\\S-1_folder' #folder where S-1.zip are stored
product_path = 'D:\\Prova'
input_S1_files = sorted(list(iglob(join(product_path, '**', '*S1*.zip'), recursive=True)))
name, sensing_mode, product_type, polarization, height, width, band_names = ([] for i in range(7))

s1_data = []  
for i in input_S1_files:
    sensing_mode.append(i.split("_")[3])
    product_type.append(i.split("_")[4])
    polarization.append(i.split("_")[-6])
    s1_read = snappy.ProductIO.readProduct(i)
    name.append(s1_read.getName())
    height.append(s1_read.getSceneRasterHeight())
    width.append(s1_read.getSceneRasterWidth())
    band_names.append(s1_read.getBandNames())
    s1_data.append(s1_read)   


# In[6]:


def read(filename):
    return ProductIO.readProduct(filename)

def write(product, filename):
    ProductIO.writeProduct(product, filename, "BEAM-DIMAP")


# In[7]:


'''APPLY ORBIT FILE'''
def apply_orbit_file(product):
    parameters = snappy.HashMap()
    parameters.put('Apply-Orbit-File', True)    
    return snappy.GPF.createProduct("Apply-Orbit-File", parameters, product)


# In[8]:


'''THERMAL NOISE REMOVAL'''
def thermal_noise_removal(product):
    parameters = snappy.HashMap()
    parameters.put('removeThermalNoise', True)
    return snappy.GPF.createProduct('thermalNoiseRemoval', parameters, product)


# In[9]:


'''CALIBRATION'''
def calibration(product):
    parameters = snappy.HashMap()
    parameters.put('selectedPolarisations', 'VH,VV')
    parameters.put('outputBetaBand', True)
    parameters.put('outputSigmaBand', False)
    parameters.put('outputImageScaleInDb', False)
    return snappy.GPF.createProduct('Calibration', parameters, product)


# In[10]:


'''TERRAIN FLATTENING'''
def terrain_flattening(product):
    parameters = snappy.HashMap()
    return snappy.GPF.createProduct('Terrain-Flattening', parameters, product)


# In[11]:


'''TERRAIN CORRECTION'''
proj_PO = '''PROJCS["WGS 84 / UTM zone 29N",
              GEOGCS["WGS 84",
              DATUM["WGS_1984",
              SPHEROID["WGS 84",6378137,298.257223563,
              AUTHORITY["EPSG","7030"]],
              AUTHORITY["EPSG","6326"]],
              PRIMEM["Greenwich",0.0,
              AUTHORITY["EPSG","8901"]],
              UNIT["degree",0.0174532925199433,
              AUTHORITY["EPSG","9122"]],
              AUTHORITY["EPSG","4326"]],
              PROJECTION["Transverse_Mercator"],
              PARAMETER["latitude_of_origin",0],
              PARAMETER["central_meridian",-9],
              PARAMETER["scale_factor",0.9996],
              PARAMETER["false_easting",500000],
              PARAMETER["false_northing",0],
              UNIT["m",1, AUTHORITY["EPSG","9001"]],
              AXIS["Easting",EAST],
              AXIS["Northing",NORTH],
              AUTHORITY["EPSG","32629"]]'''

proj_IT = '''PROJCS["WGS 84 / UTM zone 33N",
              GEOGCS["WGS 84",
              DATUM["WGS_1984",
              SPHEROID["WGS 84",6378137,298.257223563,
              AUTHORITY["EPSG","7030"]],
              AUTHORITY["EPSG","6326"]],
              PRIMEM["Greenwich",0.0,
              AUTHORITY["EPSG","8901"]],
              UNIT["degree",0.0174532925199433,
              AUTHORITY["EPSG","9122"]],
              AUTHORITY["EPSG","4326"]],
              PROJECTION["Transverse_Mercator"],
              PARAMETER["latitude_of_origin",0],
              PARAMETER["central_meridian",15],
              PARAMETER["scale_factor",0.9996],
              PARAMETER["false_easting",500000],
              PARAMETER["false_northing",0],
              UNIT["m",1, AUTHORITY["EPSG","9001"]],
              AXIS["Easting",EAST],
              AXIS["Northing",NORTH],
              AUTHORITY["EPSG","32633"]]''' 


def terrain_correction(product, proj):
    parameters = snappy.HashMap()
    parameters.put('demName', 'SRTM 1Sec HGT')
    parameters.put('pixelSpacingInMeter', 10.0)
    parameters.put('mapProjection', proj)
    parameters.put('nodataValueAtSea', False)
    return snappy.GPF.createProduct('Terrain-Correction', parameters, product)


# In[12]:


'''SUBSET'''
wkt_PO= 'POLYGON((-8.76874876510233392 37.15010193049546672, -8.76734497314194883 37.60592873585844131, -7.96268434574197403 37.60159916951047876, -7.96939235295303483 37.14584665530825447, -8.76874876510233392 37.15010193049546672))'
geom_PO = snappy.WKTReader().read(wkt_PO)

def subset(product, geom):
    parameters = snappy.HashMap()
    parameters.put('geoRegion', geom)
    parameters.put('copyMetadata', True)
    return snappy.GPF.createProduct('Subset', parameters, product)


# In[13]:


'''COREGISTRATION: CREATE STACK'''
def create_stack(product):
    parameters = snappy.HashMap()
    parameters.put('extent','Master')
    return snappy.GPF.createProduct("CreateStack", parameters, product)


# In[14]:


'''MULTITEMPORAL SPECKLE FILTER'''

def multitemporal_filter(product):
    parameters = snappy.HashMap()
    parameters.put('filter', 'Lee')
    parameters.put('filterSizeX', 10)
    parameters.put('filterSizeY', 10)
    return snappy.GPF.createProduct('Multi-Temporal-Speckle-Filter', parameters, product)


# In[28]:


'''GLCM'''
def GLCM(product):
    parameters = snappy.HashMap()
    parameters.put('windowSizeStr', '11x11')
    parameters.put('outputContrast', False)
    parameters.put('outputASM', False)
    parameters.put('outputEnergy', False)
    parameters.put('outputHomogeneity', False)
    parameters.put('outputMAX', False)
    return snappy.GPF.createProduct('GLCM', parameters, product)


# In[15]:


'''Process'''

def SAR_preprocessing_workflow(collection):
    pre_list = []
    for i in collection:
        a = apply_orbit_file(i)
        b = thermal_noise_removal(a)
        c = calibration(b)
        d = terrain_flattening(c)
        e = terrain_correction(d, proj_PO)
        f = subset(e, geom_PO)
        pre_list.append(f)
    return pre_list

PL = SAR_preprocessing_workflow(s1_data)
stack = create_stack(PL)
filtered = multitemporal_filter(stack)


# In[16]:


bands = filtered.getBandNames()
#print("Bands:%s" % (list(bands)))
bands_name= list(bands)
bands_name


# In[17]:


'''Creation of Time-Average'''
BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
targetBand1 = BandDescriptor()
targetBand1.name = 'TimeAverage_VH_PreFire'
targetBand1.type = 'float32'
targetBand1.expression = '(Gamma0_VH_mst_30Aug2018 + Gamma0_VH_slv1_18Aug2018) / 2'

targetBand2 = BandDescriptor()
targetBand2.name = 'TimeAverage_VH_PostFire'
targetBand2.type = 'float32'
targetBand2.expression = '(Gamma0_VH_mst_30Aug2018 + Gamma0_VH_slv1_18Aug2018) / 2'

targetBand3 = BandDescriptor()
targetBand3.name = 'TimeAverage_VV_PreFire'
targetBand3.type = 'float32'
targetBand3.expression = '(Gamma0_VH_mst_30Aug2018 + Gamma0_VH_slv1_18Aug2018) / 2'

targetBand4 = BandDescriptor()
targetBand4.name = 'TimeAverage_VV_PostFire'
targetBand4.type = 'float32'
targetBand4.expression = '(Gamma0_VH_mst_30Aug2018 + Gamma0_VH_slv1_18Aug2018) / 2'

targetBands = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 4)
targetBands[0] = targetBand1
targetBands[1] = targetBand2
targetBands[2] = targetBand3
targetBands[3] = targetBand4

parameters = snappy.HashMap()
parameters.put('targetBands', targetBands)

TimeAverage = GPF.createProduct('BandMaths', parameters, filtered)
    


# In[18]:


bands = TimeAverage.getBandNames()
#print("Bands:%s" % (list(bands)))
bands_name= list(bands)
bands_name


# In[25]:


'''Creation of Time-Average'''
BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
targetBand1 = BandDescriptor()
targetBand1.name = 'RBD_VH'
targetBand1.type = 'float32'
targetBand1.expression = 'TimeAverage_VH_PostFire - TimeAverage_VH_PreFire'

targetBand2 = BandDescriptor()
targetBand2.name = 'RBD_VV'
targetBand2.type = 'float32'
targetBand2.expression = 'TimeAverage_VV_PostFire - TimeAverage_VV_PreFire'

targetBand3 = BandDescriptor()
targetBand3.name = 'LogRBR_VH'
targetBand3.type = 'float32'
targetBand3.expression = 'log10(TimeAverage_VH_PostFire / TimeAverage_VH_PreFire)'

targetBand4 = BandDescriptor()
targetBand4.name = 'LogRBR_VV'
targetBand4.type = 'float32'
targetBand4.expression = 'log10(TimeAverage_VV_PostFire / TimeAverage_VV_PreFire)'

RVI_post = '4 * TimeAverage_VH_PostFire/(TimeAverage_VV_PostFire + TimeAverage_VH_PostFire)'
RVI_pre = '4 * TimeAverage_VH_PreFire/(TimeAverage_VV_PreFire + TimeAverage_VH_PreFire)'
DPSVI_post = '(TimeAverage_VV_PostFire + TimeAverage_VH_PostFire)/TimeAverage_VV_PostFire'
DPSVI_pre = '(TimeAverage_VV_PreFire + TimeAverage_VH_PreFire)/TimeAverage_VV_PreFire'

targetBand5 = BandDescriptor()
targetBand5.name = 'DeltaRVI'
targetBand5.type = 'float32'
targetBand5.expression = RVI_post + '-' + RVI_pre

targetBand6 = BandDescriptor()
targetBand6.name = 'DeltaDPSVI'
targetBand6.type = 'float32'
targetBand6.expression = DPSVI_post + '-' + DPSVI_pre

targetBands = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 6)
targetBands[0] = targetBand1
targetBands[1] = targetBand2
targetBands[2] = targetBand3
targetBands[3] = targetBand4
targetBands[4] = targetBand5
targetBands[5] = targetBand6

parameters = snappy.HashMap()
parameters.put('targetBands', targetBands)

Indices = GPF.createProduct('BandMaths', parameters, TimeAverage)


# In[26]:


bands = Indices.getBandNames()
#print("Bands:%s" % (list(bands)))
bands_name= list(bands)
bands_name


# In[30]:


'''GLCM'''
glcm = GLCM(Indices)


# In[ ]:


'''WRITING'''
write(glcm, 'C:folder\\dataset_output.dim')


# In[ ]:





# In[ ]:


from osgeo import gdal, osr
import json, re, itertools, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# In[ ]:


"Open the S-1 Dataset"

directory = r'C:folder\\dataset_output.data'

list_files = []
list_bands = []
dirFileList = os.listdir(directory)

os.chdir(directory)    
for file in dirFileList: 
    if os.path.splitext(file)[-1] == '.img': 
        list_files.append(os.path.join(directory, file))
        
for file in list_files:
    img = gdal.Open(file)
    band = img.GetRasterBand(1).ReadAsArray()
    list_bands.append(band)

list_files
list_bands


# In[ ]:


'''Reshaping'''
colxrig = list_bands[0].shape[0] * list_bands[0].shape[1]
vectors = [band.reshape(colxrig,1) for band in list_bands]     
reshapedDataset = np.array(vectors).reshape(len(list_bands),colxrig).transpose()


# In[ ]:


'''Rescale '''
from sklearn.preprocessing import MinMaxScaler

normal = MinMaxScaler()
dataset= normal.fit_transform(reshapedDataset)


# In[ ]:


'''Principal Component Analysis'''
from sklearn.decomposition import PCA

pca = PCA(n_components= reshapedDataset.shape[1]).fit(dataset)
pca_dataset = pca.transform(dataset)


# In[ ]:


'''Chose the first PCs that reaches 99% of cumulative variance'''
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
subset99 = cumulative_variance[cumulative_variance<0.99]
subset99 = np.append(subset99, cumulative_variance[cumulative_variance>=0.99][0])
n_pcs = len(subset99)
pc_to_classify = pca_dataset[:,:n_pcs]


# In[ ]:


'''Silhouette Score'''
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

k_space = list(range(2,21))
k_space

results, scores = [], []
sample = np.random.choice(range(colxrig), size=100000)
for k in k_space:
    print("Running k-means with k=%d..." % k)
    model_k = KMeans(k, init='k-means++', algorithm='full')
    model_k.fit(pc_to_classify)
    clusters_k = model_k.labels_
    results.append(clusters_k.reshape(list_bands[0].shape))
    silhouetteScore = silhouette_score(reshapedDataset[sample,:], clusters_k[sample])
    scores.append(silhouetteScore)
    print("Found %d clusters with an average Silhouette Score of %.3f" % (k, silhouetteScore))


# In[ ]:


'''K-MEAN classification'''
model = KMeans(n_clusters=7, init='k-means++', algorithm='full')   #l'algoritmo utilizzato è il k-mean++
model.fit(pc_to_classify)
clusters = model.labels_

#show the image classified
classified_image = clusters.reshape(lista_bande[0].shape) 
plt.figure()
plt.imshow(classified_image) 
plt.show()

