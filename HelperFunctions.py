import pandas as pd
from ladybug.header import Header
from ladybug.datatype.generic import GenericType
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import (
    BaseCollection,
    HourlyContinuousCollection,
    MonthlyCollection,
)
from numpy import array
import numpy as np
from typing import Dict, List
import os
from glob import glob
import xarray as xr
import matplotlib.pyplot as plt
from ladybug.datatype import TYPESDICT

#Data Conversion Functions
def header_to_string(header: Header) -> str:
    """Convert a Ladybug header object into a string.
    Args:
        header (Header):
            A Ladybug header object.
    Returns:
        str:
            A Ladybug header string."""

    return f"{header.data_type} ({header.unit})"

def header_from_string(string: str) -> Header:
    """Convert a string into a Ladybug header object.
    Args:
        string (str):
            A Ladybug header string.
    Returns:
        Header:
            A Ladybug header object."""

    str_elements = string.split(" ")

    if (len(str_elements) < 2) or ("(" not in string) or (")" not in string):
        raise ValueError(
            "The string to be converted into a LB Header must be in the format 'variable (unit)'"
        )

    str_elements = string.split(" ")
    unit = str_elements[-1].replace("(", "").replace(")", "")
    data_type = " ".join(str_elements[:-1])

    try:
        data_type = TYPESDICT[data_type.replace(" ", "")]()
    except KeyError:
        data_type = GenericType(name=data_type, unit=unit)

    return Header(data_type=data_type, unit=unit, analysis_period=AnalysisPeriod())


def to_series(collection: BaseCollection) -> pd.Series:
    """Convert a Ladybug hourlyContinuousCollection object into a Pandas Series object.
    Args:
        collection: Ladybug data collection object.
    Returns:
        pd.Series: A Pandas Series object.
    """

    index = pd.to_datetime(collection.header.analysis_period.datetimes)
    if len(collection.values) == 12:
        index = pd.date_range(f"{index[0].year}-01-01", periods=12, freq="MS")

    return pd.Series(
        data=collection.values,
        index=index,
        name=header_to_string(collection.header),
    )

def from_series(series: pd.Series) -> BaseCollection:
    """Convert a Pandas Series object into a Ladybug BaseCollection-like object.
    Args:
        series (pd.Series): A Pandas Series object.
    Returns:
        BaseCollection: A Ladybug BaseCollection-like object.
    """

    header = header_from_string(series.name)
    header.metadata["source"] = "From custom pd.Series"

    freq = pd.infer_freq(series.index)
    if freq in ["H", "h"]:
        if series.index.is_leap_year.any():
            if len(series.index) != 8784:
                raise ValueError(
                    "The number of values in the series must be 8784 for leap years."
                )
        else:
            if len(series.index) != 8760:
                raise ValueError("The series must have 8760 rows for non-leap years.")

        return HourlyContinuousCollection(
            header=header,
            values=series.values,
        )

    if freq in ["M", "MS"]:
        if len(series.index) != 12:
            raise ValueError("The series must have 12 rows for months.")

        return MonthlyCollection(
            header=header,
            values=series.values.tolist(),
            datetimes=range(1, 13),
        )

    raise ValueError("The series must be hourly or monthly.")


def get_values (
    param: str,
    scenario: str,
    lon: float,
    lat: float,
    year: int,
    data:str,
) -> array:
    """Get monthly parameter values from closest location from the Climate Model
        selectable parameters are: temp, tempmin, tempmax, rhumidity, wind, pressure,
        selectable scenarios are: ssp126, ssp245, ssp585 """
    
    scn=4
    if(scenario == "sp26"):
        scn=0
    elif (scenario =="sp45"):
        scn=1
    elif (scenario =="sp85"):
        scn=2
    else:
        raise ValueError("The selected scenario is not valid")    
       
    path="notSet"
    if (param=="temp"):
        if (data=='./GLOBAL/'):
            gcm_nc_temp = glob(f'{data}tas_day_ACCESS-CM2*.nc')
            path=f'{data}{os.path.basename(gcm_nc_temp[scn])}'
            parameter="tas"
    elif (param=="tempmin"):
        if (data=='./GLOBAL/'):
            gcm_nc_tempmin = glob(f'{data}tasmin*.nc')
            path=f'{data}{os.path.basename(gcm_nc_tempmin[scn])}'
            parameter="tasmin"
    elif (param=="tempmax"):
        if (data=='./GLOBAL/'):
            gcm_nc_tempmax = glob(f'{data}tasmax*.nc')
            path=f'{data}{os.path.basename(gcm_nc_tempmax[scn])}'
            parameter="tasmax"
    elif (param=="rhumidity"):
        if (data=='./GLOBAL/'):
            gcm_nc_rh = glob(f'{data}hurs_Amon_CanESM5*.nc')
            path=f'{data}{os.path.basename(gcm_nc_rh[scn])}'
            parameter="hurs"
    elif (param=="wind"):
        if (data=='./GLOBAL/'):
            gcm_nc_wind= glob(f'{data}sfcWind_Amon_TaiESM1*.nc')
            path=f'{data}{os.path.basename(gcm_nc_wind[scn])}'
            parameter="sfcWind"
    elif (param=="pressure"):
        if (data=='./GLOBAL/'):
            gcm_nc_press= glob(f'{data}ps_Amon_CanESM5*.nc')
            path=f'{data}{os.path.basename(gcm_nc_press[scn])}'
            parameter="ps"
    elif (param=="ccover"):
        if (data=='./GLOBAL/'):
            gcm_nc_cloud = glob(f'{data}clt_*.nc')
            path=f'{data}{os.path.basename(gcm_nc_cloud[scn])}'
            parameter="clt"
    elif (param=="radiation"):
        if (data=='./GLOBAL/'):
            gcm_nc_radiation = glob(f'{data}rsds_*.nc')
            path=f'{data}{os.path.basename(gcm_nc_radiation[scn])}'
            parameter="rsds"
    else:
        raise ValueError("The selected parameter is not valid") 
    
    ds = xr.open_dataset(path)
    values = ds[parameter]
    
    try:
        selValue = values.sel(rlon=lon, rlat=lat, method='nearest')
    except:   
        selValue = values.sel(lon=lon, lat=lat, method='nearest')
 
    selValue = selValue.sel(time=year)
    
    if (parameter=='tas'or parameter=='tasmin' or parameter=='tasmax' ):
        return selValue.values - 273.15
    else:
        return selValue.values

def _factors_to_annual(factors: List[float]) -> List[float]:
    """Cast monthly morphing factors to annual hourly ones."""
    if len(factors) != 12:
        raise ValueError(f"This method won't work ({len(factors)} != 12).")
    year_idx = pd.date_range("2050-01-01 00:00:00", freq="60min", periods=8760)
    month_idx = pd.date_range("2050-01-01 00:00:00", freq="MS", periods=12)

    # expand values across an entire year, filling NaNs where unavailable, and bookend
    annual_values_nans = (
        pd.Series(data=factors, index=month_idx).reindex(year_idx, method=None).values
    )
    annual_values_nans[-1] = annual_values_nans[0]

    # interpolate between NaNs
    return pd.Series(annual_values_nans).interpolate().values

def _daily_to_annual(factors: List[float]) -> List[float] :
    """Cast daily morphing factors to annual hourly ones."""
    if len(factors) != 365:
        raise ValueError(f"This method won't work ({len(factors)} != 365).")
    year_idx = pd.date_range("2050-01-01 00:00:00", freq="60min", periods=8760)
    day_idx = pd.date_range("2050-01-01 00:00:00", periods=365)

    # expand values across an entire year, filling NaNs where unavailable, and bookend
    annual_values_nans = (
        pd.Series(data=factors, index=day_idx).reindex(year_idx, method=None).values
    )
    annual_values_nans[-1] = annual_values_nans[0]

    # interpolate between NaNs
    return pd.Series(annual_values_nans).interpolate().values

def _plot_projection(
    dbt_new: HourlyContinuousCollection,
    series,
    title: str,
):
    plt.plot(np.array(range(0,8760 )), dbt_new.values, color="red", label="Projected Values")
    plt.plot(np.array(range(0,8760 )), series.values, color="blue", label="Original Values", alpha=0.4)
    plt.xticks(ticks=np.arange(0,8760,31*24) ,labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug','Sep','Oct', 'Nov', 'Dec'])
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()