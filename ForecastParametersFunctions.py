from HelperFunctions import header_to_string, header_from_string, to_series, from_series, get_values, _factors_to_annual, _daily_to_annual, _plot_projection
from ladybug.epw import EPW, HourlyContinuousCollection, Location, MonthlyCollection
from typing import Dict, List
from pathlib import Path
from ladybug.datatype.energyflux import (
    DiffuseHorizontalRadiation,
    DirectNormalRadiation,
    GlobalHorizontalRadiation,
)
from ladybug.datatype.fraction import OpaqueSkyCover, RelativeHumidity, TotalSkyCover
from ladybug.datatype.illuminance import (
    DiffuseHorizontalIlluminance,
    DirectNormalIlluminance,
    GlobalHorizontalIlluminance,
)
from ladybug.datatype.luminance import ZenithLuminance
from ladybug.datatype import TYPESDICT
from ladybug.datatype.generic import GenericType
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datatype.pressure import AtmosphericStationPressure
from ladybug.datatype.speed import WindSpeed
from ladybug.datatype.temperature import DryBulbTemperature
from ladybug.epw import EPW, HourlyContinuousCollection, Location, MonthlyCollection
from ladybug.psychrometrics import dew_point_from_db_rh
from ladybug.skymodel import calc_horizontal_infrared
import pandas as pd
import numpy as np
import os
import xarray as xr
import warnings

def _forecast_dry_bulb_temperature(
    location: Location,
    dbt_collection: HourlyContinuousCollection,
    scenario: str,
    year: str,
    DATADIR: str,
) -> HourlyContinuousCollection:
    """Forecast dry bulb temperature using downloaded datasets from CDS."""

    if not isinstance(dbt_collection.header.data_type, DryBulbTemperature):
        raise ValueError(
            f"This method can only forecast for dtype of {DryBulbTemperature}"
        )

    # test for data validity and return the original collection is all "invalid"
    if all(i == 99.9 for i in dbt_collection):
        warnings.warn(
            "The original dry bulb temperature values are all missing. The original data will be returned instead."
        )
        return dbt_collection

    # attempt to transform the input data
    index = pd.to_datetime(dbt_collection.header.analysis_period.datetimes)
    
    #values from GCM  
    tmin = _daily_to_annual(
        get_values( param = "tempmin",scenario = scenario,lon=location.longitude,lat= location.latitude, year = year, data=DATADIR)
    )
    tmin_series = pd.Series(tmin, index=index)
    temp = _daily_to_annual(
        get_values( param = "temp",scenario = scenario,lon=location.longitude,lat= location.latitude, year = year, data=DATADIR)
    )
    temp_series=  pd.Series(temp, index=index)
    tmax = _daily_to_annual(
        get_values( param = "tempmax",scenario = scenario,lon=location.longitude,lat= location.latitude, year = year, data=DATADIR)
    )
    tmax_series = pd.Series(tmax, index=index)
    dbt_new_monthly_mean = (
        temp_series.resample("1D")
        .mean()
        .resample("MS")
        .mean()
        .reindex(temp_series.index, method="ffill")
    )
    dbt_new_monthly_min = (
        tmin_series.resample("1D")
        .mean()
        .resample("MS")
        .mean()
        .reindex(tmin_series.index, method="ffill")
    )
    dbt_new_monthly_max = (
        tmax_series.resample("1D")
        .mean()
        .resample("MS")
        .mean()
        .reindex(tmax_series.index, method="ffill")
    )
    #values from epw
    series = to_series(dbt_collection)
    dbt_0_monthly_average_daily_max = (
        series.resample("1D")
        .max()
        .resample("MS")
        .mean()
        .reindex(series.index, method="ffill")
    )
    dbt_0_monthly_average_daily_mean = (
        series.resample("MS").mean().reindex(series.index, method="ffill")
    )
    dbt_0_monthly_average_daily_min = (
        series.resample("1D")
        .min()
        .resample("MS")
        .mean()
        .reindex(series.index, method="ffill")
    )
    Dtemp= dbt_new_monthly_mean-dbt_0_monthly_average_daily_mean
    Dtmax= dbt_new_monthly_max-dbt_0_monthly_average_daily_max
    Dtmin= dbt_new_monthly_min-dbt_0_monthly_average_daily_min
    
    adbt_m = (Dtmax - Dtmin) / (
        dbt_0_monthly_average_daily_max - dbt_0_monthly_average_daily_min
    )
    
    
    dbt_new = series + Dtemp + adbt_m * (series - dbt_0_monthly_average_daily_mean)

    _plot_projection(dbt_new, series, title="Projected Changes in Temperature")
    
    # last check to ensure results arent weird
    avg_diff_limit = 20
    if not np.allclose(series, dbt_new, atol=avg_diff_limit):
        warnings.warn(
            "Forecast for dry-bulb temperature returns values beyond feasible range of transformation. The original data will be returned instead."
        )
        return dbt_collection
    dbt_new.name=series.name
    return from_series(dbt_new)
def _forecast_relative_humidity(
    location: Location,
    rh_collection: HourlyContinuousCollection,
    scenario: str,
    year: int,
    DATADIR: str,
) -> HourlyContinuousCollection:
    """Forecast relative humidity using downloaded datasets from CDS."""

    if not isinstance(rh_collection.header.data_type, RelativeHumidity):
        raise ValueError(
            f"This method can only forecast for dtype of {RelativeHumidity}"
        )

    # test for data validity and return the original collection is all "invalid"
    if all(i == 999 for i in rh_collection):
        warnings.warn(
            "The original relative humidity values are all missing. The original data will be returned instead."
        )
        return rh_collection

    # attempt to transform the input data
    series = to_series(rh_collection)
    
    rh_0_monthly_average_daily_mean = (
        series.resample("MS").mean().reindex(series.index, method="ffill")
    )
    rhum = _factors_to_annual(
        get_values( param = "rhumidity",scenario = scenario,lon=location.longitude,lat= location.latitude, year = year, data=DATADIR)
    )
    rh_series = pd.Series(rhum, index=pd.to_datetime(rh_collection.header.analysis_period.datetimes))
    rh_fact = (rh_series-rh_0_monthly_average_daily_mean)/rh_0_monthly_average_daily_mean
    rh_new = (series*(1+rh_fact)).clip(0, 100)

    _plot_projection(rh_new, series, title="Projected Changes in Relative Humidity")

    # last check to ensure results arent weird
    avg_diff_limit = 20
    if not np.allclose(series, rh_new, atol=avg_diff_limit):
        warnings.warn(
            "Forecast for relative humidity returns values beyond feasible range of transformation. The original data will be returned instead."
        )
        return rh_collection
    rh_new.name=series.name
    return from_series(rh_new)
    
def _forecast_atmospheric_pressure(
    location: Location,
    ap_collection: HourlyContinuousCollection,
    scenario: str,
    year: int,
    DATADIR: str,
) -> HourlyContinuousCollection:
    """Forecast atmospheric pressure using downloaded datasets from CDS."""

    if not isinstance(ap_collection.header.data_type, AtmosphericStationPressure):
        raise ValueError(
            f"This method can only forecast for dtype of {AtmosphericStationPressure}"
        )

    # test for data validity and return the original collection is all "invalid"
    if all(i == 999999 for i in ap_collection):
        warnings.warn(
            "The original atmospheric pressure values are all missing. The original data will be returned instead."
        )
        return ap_collection

    # attempt to transform the input data
    series = to_series(ap_collection)
    ap_0_monthly_average_daily_mean = (
        series.resample("MS").mean().reindex(series.index, method="ffill")
    )
    mslp = _factors_to_annual(
        get_values( param = "pressure",scenario = scenario,lon=location.longitude,lat= location.latitude, year = year, data=DATADIR)
    )
    ap_series = pd.Series(mslp, index=pd.to_datetime(ap_collection.header.analysis_period.datetimes))
    ap_new = series + (ap_series-ap_0_monthly_average_daily_mean)

    _plot_projection(ap_new, series, title="Projected Changes in Atmospheric Pressure")
    
    # last check to ensure results arent weird
    avg_diff_limit = 300
    if not np.allclose(series, ap_new, atol=avg_diff_limit):
        warnings.warn(
            "Forecast for atmospheric pressure returns values beyond feasible range of transformation. The original data will be returned instead."
        )
        return ap_collection
    ap_new.name=series.name
    return from_series(ap_new)
    
def _calculate_dew_point_temperature(
    dbt_collection: HourlyContinuousCollection,
    rh_collection: HourlyContinuousCollection,
) -> HourlyContinuousCollection:
    """Calculate DPT from composite variables."""

    if all(i == 99.9 for i in dbt_collection) or all(i == 999 for i in rh_collection):
        warnings.warn(
            "The original dry bulb temperature or relative humidity values are all missing. Dew point temperature will be constructed from the default value for missing values."
        )
        return EPW.from_missing_values().dew_point_temperature

    _dbt = to_series(dbt_collection)
    _rh = to_series(rh_collection)

    dpt = []
    for dbt, rh in list(zip(*[_dbt, _rh])):
        dpt.append(dew_point_from_db_rh(dbt, rh))
    freq = pd.infer_freq(_dbt.index)
    
    
    return from_series(
        pd.Series(dpt, index=_dbt.index, name="Dew Point Temperature (C)")
    )

def _forecast_wind_speed(
    location: Location,
    ws_collection: HourlyContinuousCollection,
    scenario: str,
    year: int,
    DATADIR: str,
) -> HourlyContinuousCollection:
    """Forecast wind speed using downloaded datasets from CDS."""

    if not isinstance(ws_collection.header.data_type, WindSpeed):
        raise ValueError(f"This method can only forecast for dtype of {WindSpeed}")

    # test for data validity and return the original collection is all "invalid"
    if all(i == 999 for i in ws_collection):
        warnings.warn(
            "The original wind speed values are all missing. The original data will be returned instead."
        )
        return ws_collection

    # attempt to transform the input data
    series = to_series(ws_collection)
    wind = _factors_to_annual(
       get_values( param = "wind",scenario = scenario,lon=location.longitude,lat= location.latitude, year = year, data=DATADIR)
    )
    w_0_monthly_average_daily_mean = (
        series.resample("MS").mean().reindex(series.index, method="ffill")
    )
    w_series = pd.Series(wind, index=pd.to_datetime(ws_collection.header.analysis_period.datetimes))
    ws_new = (1 + (w_series-w_0_monthly_average_daily_mean)/w_0_monthly_average_daily_mean)*series

    _plot_projection(ws_new, series, title="Projected Changes in Wind Speed")
  
    # last check to ensure results arent weird
    avg_diff_limit = 10
    if not np.allclose(series, ws_new, atol=avg_diff_limit):
        warnings.warn(
            "Forecast for wind speed returns values beyond feasible range of transformation. The original data will be returned instead."
        )
        return ws_collection
    ws_new.name=series.name
    return from_series(ws_new)

def _forecast_sky_cover(
    location: Location,
    sc_collection: HourlyContinuousCollection,
    scenario: str,
    year: int,
    DATADIR: str,
) -> HourlyContinuousCollection:
    """Forecast sky cover using IPCC HadCM3 forecast model."""

    if not isinstance(sc_collection.header.data_type, (TotalSkyCover, OpaqueSkyCover)):
        raise ValueError(
            f"This method can only forecast for dtypes of {TotalSkyCover, OpaqueSkyCover}"
        )

    # test for data validity and return the original collection is all "invalid"
    if all(i == 99 for i in sc_collection):
        warnings.warn(
            "The original sky cover values are all missing. The original data will be returned instead."
        )
        return sc_collection


    series = to_series(sc_collection)
    ccov = _factors_to_annual(
       get_values( param = "ccover",scenario = scenario,lon=location.longitude,lat= location.latitude, year = year, data=DATADIR)
    )
    cc_0_monthly_average_daily_mean = (
        series.resample("MS").mean().reindex(series.index, method="ffill")
    )
    cc_series = pd.Series(ccov, index=pd.to_datetime(sc_collection.header.analysis_period.datetimes))
    Dccov= cc_series/10-cc_0_monthly_average_daily_mean
    sc_new = (series + Dccov).clip(0, 10)

    _plot_projection(sc_new, series, title="Projected Changes in Cloud Cover")
    
    # last check to ensure results arent weird
    avg_diff_limit = 10
    if not np.allclose(series, sc_new, atol=avg_diff_limit):
        warnings.warn(
            "Forecast for sky cover returns values beyond feasible range of transformation. The original data will be returned instead."
        )
        return sc_collection
        
    sc_new.name=series.name
    return from_series(sc_new)
    
def _calculate_horizontal_infrared_radiation_intensity(
    osc_collection: HourlyContinuousCollection,
    dbt_collection: HourlyContinuousCollection,
    dpt_collection: HourlyContinuousCollection,
) -> HourlyContinuousCollection:
    """Calculate HIR from composite variables."""

    if (
        all(i == 99.9 for i in dbt_collection)
        or all(i == 99 for i in osc_collection)
        or all(i == 99.9 for i in dpt_collection)
    ):
        warnings.warn(
            "The original OSC, DBT or DPT values are all missing. HIR will be constructed from the default value for missing values."
        )
        return EPW.from_missing_values().horizontal_infrared_radiation_intensity

    _osc = to_series(osc_collection)
    _dbt = to_series(dbt_collection)
    _dpt = to_series(dpt_collection)

    hir = []
    for osc, dbt, dpt in list(zip(*[_osc, _dbt, _dpt])):
        hir.append(calc_horizontal_infrared(osc, dbt, dpt))

    return from_series(
        pd.Series(
            hir, index=_dbt.index, name="Horizontal Infrared Radiation Intensity (W/m2)"
        )
    )

def _calculate_radiation_factor(
    location: Location,
    solar_collection: HourlyContinuousCollection,
    scenario: str,
    year: int,
    DATADIR: str,
) -> HourlyContinuousCollection:
    """Forecast solar variables using IPCC HadCM3 forecast model."""
# attempt to transform the input data
    series = to_series(solar_collection)
    sol = _factors_to_annual(
       get_values( param = "radiation",scenario = scenario,lon=location.longitude,lat= location.latitude, year = year, data=DATADIR)
    )
    sol_0_monthly_average = (
        series.resample("MS").mean().reindex(series.index, method="ffill")
    )
    sol_series = pd.Series(sol, index=pd.to_datetime(solar_collection.header.analysis_period.datetimes))
    solar_factor = 1 + (sol-sol_0_monthly_average)/sol_0_monthly_average
    solar_factor.iloc[-1] = solar_factor.iloc[0]
    return solar_factor 


def _forecast_solar(
    location: Location,
    solar_collection: HourlyContinuousCollection,
    scenario: str,
    year: int,
    solar_factor: float,
    DATADIR: str,
) -> HourlyContinuousCollection:
    """Forecast solar variables using IPCC HadCM3 forecast model."""

    if not isinstance(
        solar_collection.header.data_type,
        (
            GlobalHorizontalRadiation,
            GlobalHorizontalIlluminance,
            DirectNormalRadiation,
            DirectNormalIlluminance,
            DiffuseHorizontalRadiation,
            DiffuseHorizontalIlluminance,
            ZenithLuminance,
        ),
    ):
        raise ValueError(
            f"This method can only forecast for dtypes of {GlobalHorizontalRadiation, GlobalHorizontalIlluminance, DirectNormalRadiation, DirectNormalIlluminance, DiffuseHorizontalRadiation, DiffuseHorizontalIlluminance, ZenithLuminance}, not {type(solar_collection.header.data_type)}"
        )

    # test for data validity and return the original collection is all "invalid"
    if isinstance(
        solar_collection.header.data_type,
        (
            GlobalHorizontalRadiation,
            DirectNormalRadiation,
            DiffuseHorizontalRadiation,
            ZenithLuminance,
        ),
    ):
        missing_val = 9999
    else:
        missing_val = 999999
    if all(i == missing_val for i in solar_collection):
        warnings.warn(
            "The original solar values are all missing. The original data will be returned instead."
        )
        return solar_collection

    # attempt to transform the input data
    series = to_series(solar_collection)
    sc_new = (series * solar_factor.interpolate()).clip(lower=0)

    _plot_projection(sc_new, series, title="Projected Changes in Solar Radiation Values")

    # last check to ensure results arent weird
    avg_diff_limit = 200
    if not np.allclose(series, sc_new, atol=avg_diff_limit):
        warnings.warn(
            "Forecast for solar values returns values beyond feasible range of transformation. The original data will be returned instead."
        )
        return solar_collection
    sc_new.name=series.name
    return from_series(sc_new)

def _modify_ground_temperature(
    original_epw: EPW, new_epw: EPW
) -> Dict[str, MonthlyCollection]:
    """Based on changes in DBT from a source and target EPW file, adjust the source monthly ground temperatures accordingly.
    Args:
        original_epw (EPW):
            The source EPW file.
        new_epw (EPW):
            The target EPW file.
    Returns:
        Dict[str, MonthlyCollection]:
            A set of Monthly ground temperature data collections.
    """
    factors = (
        to_series(new_epw.dry_bulb_temperature).resample("MS").mean()
        / to_series(original_epw.dry_bulb_temperature).resample("MS").mean()
    ).values
    new_ground_temperatures = {}
    for depth, collection in original_epw.monthly_ground_temperature.items():
        new_ground_temperatures[depth] = MonthlyCollection(
            header=collection.header,
            values=factors * collection.values,
            datetimes=collection.datetimes,
        )
    return new_ground_temperatures

def forecast_epw(epw: EPW, emissions_scenario: str, forecast_year: int, datadir:str) -> EPW:
    """Forecast an EPW using the methodology provided by Belcher et al, "Constructing design weather data for future climates"
    Args:
        epw (EPW):
            The EPW file to transform.
        emissions_scenario (str):
            An emissions scenario to forecast with.
        forecast_year (int):
            The year to forecast.
    Returns:
        EPW:
            A "forecast" EPW file.
    """


    # create an "empty" epw object eready to populate
    new_epw = EPW.from_missing_values(epw.is_leap_year)
    new_epw.location = epw.location
    new_epw.comments_1 = f"{epw.comments_1}. Forecast using transformation factors from the IPCC HadCM3 {emissions_scenario} emissions scenario for {forecast_year} according to the methodology from Jentsch M.F., James P.A.B., Bourikas L. and Bahaj A.S. (2013) Transforming existing weather data for worldwide locations to enable energy and building performance simulation under future climates, Renewable Energy, Volume 55, pp 514-524."
    new_epw.comments_2 = epw.comments_2
    new_epw._file_path = (
        Path(epw.file_path).parent
        / f"{Path(epw.file_path).stem}__CDS_{emissions_scenario}_{forecast_year}.epw"
    ).as_posix()

    # copy over variables that aren't going to change
    new_epw.years.values = epw.years.values
    new_epw.wind_direction.values = epw.wind_direction.values
    new_epw.present_weather_observation.values = epw.present_weather_observation.values
    new_epw.present_weather_codes.values = epw.present_weather_codes.values
    new_epw.aerosol_optical_depth.values = epw.aerosol_optical_depth.values
    new_epw.snow_depth.values = epw.snow_depth.values
    new_epw.days_since_last_snowfall.values = epw.days_since_last_snowfall.values
    new_epw.albedo.values = epw.albedo.values
    new_epw.liquid_precipitation_depth.values = epw.liquid_precipitation_depth.values
    new_epw.liquid_precipitation_quantity.values = (
        epw.liquid_precipitation_quantity.values
    )
    new_epw.precipitable_water.values = epw.precipitable_water.values

    # forecast variables
    new_epw.dry_bulb_temperature.values = _forecast_dry_bulb_temperature(
        epw.location, epw.dry_bulb_temperature, emissions_scenario, forecast_year, datadir
    ).values
    new_epw.relative_humidity.values = _forecast_relative_humidity(
        epw.location, epw.relative_humidity, emissions_scenario, forecast_year, datadir
    ).values
    new_epw.dew_point_temperature.values = _calculate_dew_point_temperature(
        new_epw.dry_bulb_temperature, new_epw.relative_humidity
    ).values
    new_epw.wind_speed.values = _forecast_wind_speed(
        epw.location, epw.wind_speed, emissions_scenario, forecast_year, datadir
    ).values
    try:
        new_epw.atmospheric_station_pressure.values = _forecast_atmospheric_pressure(
            epw.location,
            epw.atmospheric_station_pressure,
            emissions_scenario,
            forecast_year,
            datadir
        )
    except:
        new_epw.atmospheric_station_pressure.values = epw.atmospheric_station_pressure
   
    new_epw.total_sky_cover.values = _forecast_sky_cover(
        epw.location, epw.total_sky_cover, emissions_scenario, forecast_year, datadir
    ).values
    new_epw.opaque_sky_cover.values = _forecast_sky_cover(
        epw.location, epw.opaque_sky_cover, emissions_scenario, forecast_year, datadir
    ).values
   
    new_epw.horizontal_infrared_radiation_intensity.values = (
    _calculate_horizontal_infrared_radiation_intensity(
        new_epw.opaque_sky_cover,
        new_epw.dry_bulb_temperature,
        new_epw.dew_point_temperature,
    ).values
    )
    
    solar_factor=_calculate_radiation_factor(epw.location, epw.global_horizontal_radiation, emissions_scenario, forecast_year, datadir)
    
    new_epw.global_horizontal_radiation.values = _forecast_solar(
        epw.location, epw.global_horizontal_radiation, emissions_scenario, forecast_year, solar_factor, datadir
    ).values
    new_epw.direct_normal_radiation.values = _forecast_solar(
        epw.location, epw.direct_normal_radiation, emissions_scenario, forecast_year, solar_factor, datadir
    ).values
    new_epw.diffuse_horizontal_radiation.values = _forecast_solar(
        epw.location, epw.diffuse_horizontal_radiation, emissions_scenario, forecast_year, solar_factor, datadir
    ).values
    new_epw.global_horizontal_illuminance.values = _forecast_solar(
        epw.location, epw.global_horizontal_illuminance, emissions_scenario, forecast_year, solar_factor, datadir
    ).values
    new_epw.direct_normal_illuminance.values = _forecast_solar(
        epw.location, epw.direct_normal_illuminance, emissions_scenario, forecast_year, solar_factor, datadir
    ).values
    new_epw.diffuse_horizontal_illuminance.values = _forecast_solar(
        epw.location, epw.diffuse_horizontal_illuminance, emissions_scenario, forecast_year,  solar_factor, datadir
    ).values
    new_epw.zenith_luminance.values = _forecast_solar(
        epw.location, epw.zenith_luminance, emissions_scenario, forecast_year,  solar_factor, datadir
    ).values
    # modify ground temperatures based on differences in EPW DBT
    new_epw._monthly_ground_temps = _modify_ground_temperature(epw, new_epw)

    return new_epw