import xarray as xr 
import os
DATA_PATH = os.environ.get("DATA_PATH")

ds = xr.open_dataset(os.path.join(DATA_PATH,"ww3.gom.2022030612.nc"))
print(list(ds.keys()))

hs = xr.where(ds['phs1'].isnull(),1,0)
hs0 = xr.where(ds['phs0'].isnull(),-1,0)
s = hs+hs0
print(s.sum())

with xr.open_dataset(os.path.join(DATA_PATH,"ww3.gom.2022030612.nc")) as ds:
    print(ds.dims)
