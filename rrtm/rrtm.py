# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:15:38 2016

@author: mcgibbon
"""
import librrtm_lw_wrapper
import librrtm_sw_wrapper
import chemistry
import numpy as np


def mean_air_temperature_from_interfaces(
        interface_air_temperature, interface_air_pressure):
    """
    Returns a mass-weighted layer average air temperature, assuming that
    temperature is linearly interpolated in log-pressure coordinates.
    """
    T = interface_air_temperature
    p = interface_air_pressure
    log_p_ratio = np.log(p[1:]/p[:-1])
    return (
        T[:-1] - (T[1:] - T[:-1]) / ((p[1:] - p[:-1]) * - log_p_ratio) *
        (p[1:] * log_p_ratio - p[1:] + p[:-1])
    )


def mean_air_pressure_from_interfaces(interface_air_pressure):
    """Returns the mass-weighted layer average pressure."""
    return (interface_air_pressure[1:] + interface_air_pressure[:-1]) / 2.


def generate_wkl_wbrodl(self):
    # first create and fill wkl
    wkl = np.zeros((len(_species_names), self._nlayers), dtype="double")
    for i, s in enumerate(_species_names):
        if s in self._species_vmr:
            wkl[i] = self._species_vmr[s]
    # next generate wbrodl
    coldry = chemistry.column_density_rrtm(self.tavel, self.pz, self.pavel)
    if (wkl < 1.).all():
        wbrodl = coldry * (1 - wkl[1:].sum(0))
    elif (wkl > 1.).all():
        wbrodl = coldry - wkl[1:].sum(0)
    else:
        raise ValueError("WKL units issue detected.")
    return wkl, wbrodl


class RRTMError(Exception):
    pass


def get_species_volume_mixing_ratio_array(volume_mixing_ratios, num_levels):
    """
    Takes in a dictionary whose keys are strings indicating gas species and values are
    1-D (vertical) volume mixing ratio arrays of those gases. Also takes in the number of
    vertical levels. Returns a single 2-D array whose first axis is the vertical axis and
    second axis is species. This array is the wkl array required by the RRTM code.
    """
    gas_names = ('h2o', 'co2', 'o3', 'n2o', 'co', 'ch4', 'o2')
    return_array = np.zeros((num_levels, len(gas_names)))
    for i, name in enumerate(gas_names):
        if name in volume_mixing_ratios:
            return_array[:, i] = volume_mixing_ratios[name]
    return return_array


def get_broadening_volume_mixing_ratio_array(
        species_volume_mixing_ratio_array, mean_air_temperature, interface_air_pressure,
        mean_air_pressure):
    """
    Takes in a 2-D species volume mixing ratio array (as from
    get_species_volume_mixing_ratio_array), and 1-D arrays for mass mean air temperature,
    interface air pressure, and mass-mean air pressure. Returns the 1-D (height) array
    for volume mixing ratio of broadening gases. This is the wbroadl array required by
    the RRTM code.
    """
    coldry = chemistry.column_density_rrtm(
        mean_air_temperature, interface_air_pressure, mean_air_pressure)
    wkl = species_volume_mixing_ratio_array
    if (wkl < 1.).all():
        return_array = coldry * (1 - wkl[:, 1:].sum(axis=1))
    elif (wkl > 1.).all():
        return_array = coldry - wkl[:, 1:].sum(axis=1)
    else:
        raise ValueError("WKL units issue detected.")
    return return_array


def run_lw_rrtm(
        interface_air_temperature, interface_air_pressure, gas_volume_mixing_ratios,
        mean_air_temperature=None,
        mean_air_pressure=None, scattering='none',
        reflection='lambertian', num_angles=2, num_streams=2, surface_emissivity=1.0,
        surface_temperature=None,):
    """
    Runs the longwave RRTM code.

    Args:
        interface_air_temperature (1-d array): Air temperature at interface levels in
            Kelvin, in ascending order.
        interface_air_pressure (1-d array): Air pressure at interface levels in hPa, in
            ascending order.
        gas_volume_mixing_ratios (dict): Dictionary whose keys are strings representing
            atmospheric gases, and values are 1-D arrays containing volume mixing ratios
            of those gases at interface levels, in ascending order. Floats may also be
            used for constant mixing ratio at all levels. Considered gases are
            'H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', and 'O2'. Any gases not included will
            be set to 0.
        mean_air_temperature (1-d array, optional): Mass-weighted layer average air
            temperature in Kelvin, in ascending order. By default will be calculated from
            interface values assuming linear interpolation.
        mean_air_pressure (1-d array, optional): Mass-weighted layer average air pressure
            in hPa, in ascending order. By default will be calculated from interface
            values assuming linear interpolation.
        scattering (str, optional): Type of scattering to enable in the radiation scheme.
            'none' for no scattering. 'disort' for no scattering, but the calculation is
            performed using the DISORT code. 'yes' includes scattering but does not do
            anything as we have not allowed aerosols or clouds. Default is 'none'.
        reflection (str, optional): Reflection type. Can be 'lambertian' for Lambertian
            reflection, or 'specular' for specular reflection (where angle is equal to
            downwelling angle). Defaults to 'lambertian'.
        num_angles (int, optional):
            Number of angles used by the radiation scheme as quadriture points. Only used
            if scattering='none'. Default is 2.
        num_streams (int, optional): Number of streams used by the radiation scheme. Only
            used if scattering='disort'.
        surface_emissivity (float, optional): Surface Emissivity. 0.0 corresponds to
            no longwave emission from the surface. Defaults to 1.0.
        surface_temperature (float, optional): The surface temperature in Kelvin. Defaults
            to interface_air_temperature[0].

    Returns:
        output (dict): A dictionary containing longwave radiative fluxes and heating rate.
    """
    if mean_air_temperature is None:
        mean_air_temperature = mean_air_temperature_from_interfaces(
            interface_air_temperature, interface_air_pressure)
    if mean_air_pressure is None:
        mean_air_pressure = mean_air_pressure_from_interfaces(interface_air_pressure)
    if surface_temperature is None:
        surface_temperature = interface_air_temperature[0]
    for name, value in gas_volume_mixing_ratios.items():
        gas_volume_mixing_ratios[name.lower()] = value

    ireflect = {'lambertian': 0, 'specular': 1}[reflection.lower()]
    iscat = {'none': 0, 'disort': 1, 'yes': 2}[scattering.lower()]
    numangs = {'none': num_angles, 'disort': num_streams, 'yes': num_streams}[scattering.lower()]
    species_volume_mixing_ratio_array = get_species_volume_mixing_ratio_array(
        gas_volume_mixing_ratios, len(mean_air_pressure))
    broadening_volume_mixing_ratio_array = get_broadening_volume_mixing_ratio_array(
        species_volume_mixing_ratio_array, mean_air_temperature, interface_air_pressure,
        mean_air_pressure)

    try:
        return_values = librrtm_lw_wrapper.run_rrtm(
            iscat, numangs, surface_temperature, ireflect, surface_emissivity,
            mean_air_temperature, mean_air_pressure, interface_air_temperature,
            interface_air_pressure, species_volume_mixing_ratio_array,
            broadening_volume_mixing_ratio_array)
    except librrtm_lw_wrapper.LibRRTMError as e:
        raise RRTMError(e.data)
    return {
        'upward_longwave_flux': return_values[0],
        'downward_longwave_flux': return_values[1],
        'net_longwave_flux': return_values[2],
        'longwave_heating_rate': return_values[3]}


def run_sw_rrtm(
        interface_air_temperature, interface_air_pressure, gas_volume_mixing_ratios,
        mean_air_temperature=None,
        mean_air_pressure=None,
        reflection='lambertian',
        num_streams=2,
        surface_emissivity=1.0,
        julian_day=None,
        solar_scaling_factor=1.0,
        solar_zenith_angle=0.):
    """
    Runs the shortwave RRTM code.

    Args:
        interface_air_temperature (1-d array): Air temperature at interface levels in
            Kelvin, in ascending order.
        interface_air_pressure (1-d array): Air pressure at interface levels in hPa, in
            ascending order.
        gas_volume_mixing_ratios (dict): Dictionary whose keys are strings representing
            atmospheric gases, and values are 1-D arrays containing volume mixing ratios
            of those gases at interface levels, in ascending order. Floats may also be
            used for constant mixing ratio at all levels. Considered gases are
            'H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', and 'O2'. Any gases not included will
            be set to 0.
        mean_air_temperature (1-d array, optional): Mass-weighted layer average air
            temperature in Kelvin, in ascending order. By default will be calculated from
            interface values assuming linear interpolation.
        mean_air_pressure (1-d array, optional): Mass-weighted layer average air pressure
            in hPa, in ascending order. By default will be calculated from interface
            values assuming linear interpolation.
        reflection (str, optional): Reflection type. Can be 'lambertian' for Lambertian
            reflection, or 'specular' for specular reflection (where angle is equal to
            downwelling angle). Defaults to 'lambertian'.
        num_streams (int, optional): Number of streams used by the radiation scheme.
        surface_emissivity (float, optional): Surface Emissivity. 0.0 corresponds to
            no longwave emission from the surface. Defaults to 1.0.
        julian_day (float, optional): Julian day of the year, from 1 to 365. Used only to
            calculate the Earth-Sun distance. By default, uses an average Earth-Sun
            distance.
        solar_scaling_factor (float, optional): Factor by which to rescale the solar
            source. Defaults to 1.0.
        solar_zenith_angle (float, optional): Solar zenith angle in degrees. Defaults to
            0 degrees (overhead).

    Returns:
        output (dict): A dictionary containing the solar fluxes and radiative heating
            rate.
    """
    if mean_air_temperature is None:
        mean_air_temperature = mean_air_temperature_from_interfaces(
            interface_air_temperature, interface_air_pressure)
    if mean_air_pressure is None:
        mean_air_pressure = mean_air_pressure_from_interfaces(interface_air_pressure)
    if julian_day is None:
        julian_day = 0.  # default understood by wrapped code

    ireflect = {'lambertian': 0, 'specular': 1}[reflection.lower()]
    species_volume_mixing_ratio_array = get_species_volume_mixing_ratio_array(
        gas_volume_mixing_ratios, len(mean_air_pressure))
    broadening_volume_mixing_ratio_array = get_broadening_volume_mixing_ratio_array(
        species_volume_mixing_ratio_array, mean_air_temperature, interface_air_pressure,
        mean_air_pressure)
    try:
        return_values = librrtm_sw_wrapper.run_rrtm_sw(
            num_streams, julian_day, solar_zenith_angle, solar_scaling_factor, ireflect,
            surface_emissivity, mean_air_temperature, mean_air_pressure,
            interface_air_temperature, interface_air_pressure,
            species_volume_mixing_ratio_array,
            broadening_volume_mixing_ratio_array)
    except librrtm_sw_wrapper.LibRRTMSWError as e:
        raise RRTMError(e.data)
    return {
        'upward_shortwave_flux': return_values[0],
        'downward_shortwave_flux': return_values[1],
        'net_shortwave_flux': return_values[2],
        'shortwave_heating_rate': return_values[3],
    }
