import rrtm
import pytest
import numpy as np
import scipy.io.netcdf


def get_test_data():
    nc = scipy.io.netcdf.netcdf_file('tests/tests_data.nc', 'r')
    input_data = {
        'mean_air_pressure': nc.variables['pavel'][:].copy(),
        'mean_air_temperature': nc.variables['tavel'][:].copy(),
        'interface_air_pressure': nc.variables['pz'][:].copy(),
        'interface_air_temperature': nc.variables['tz'][:].copy(),
        'gas_volume_mixing_ratios': {
            'h2o': nc.variables['h2o'][:].copy(),
            'co2': nc.variables['co2'][:].copy() ,
            'o3': nc.variables['o3'][:].copy(),
            'n2o': nc.variables['n2o'][:].copy(),
            'co': nc.variables['co'][:].copy(),
            'ch4': nc.variables['ch4'][:].copy(),
            'o2': nc.variables['o2'][:].copy(),
        }
    }
    output_data = {
        'htr_' + name: nc.variables['htr_' + name][:].copy() for name in (
            'lw', 'lw_disort', 'lw_auto_pz', 'sw')
    }
    nc.close()
    return input_data, output_data


def test_longwave():
    input_data, output_data = get_test_data()
    model_output = rrtm.run_lw_rrtm(
        input_data['interface_air_temperature'],
        input_data['interface_air_pressure'],
        input_data['gas_volume_mixing_ratios'],
        input_data['mean_air_temperature'],
        input_data['mean_air_pressure'],
        scattering='none',
        reflection='lambertian',
        num_angles=2,
        num_streams=2,
        surface_emissivity=1.0,
        surface_temperature=None,
    )
    assert np.allclose(model_output['longwave_heating_rate'], output_data['htr_lw'])


def test_longwave_disort():
    input_data, output_data = get_test_data()
    model_output = rrtm.run_lw_rrtm(
        input_data['interface_air_temperature'],
        input_data['interface_air_pressure'],
        input_data['gas_volume_mixing_ratios'],
        input_data['mean_air_temperature'],
        input_data['mean_air_pressure'],
        scattering='disort',
        reflection='lambertian',
        num_angles=2,
        num_streams=2,
        surface_emissivity=1.0,
        surface_temperature=None,
    )
    assert np.allclose(model_output['longwave_heating_rate'], output_data['htr_lw_disort'])


def test_shortwave():
    input_data, output_data = get_test_data()
    model_output = rrtm.run_sw_rrtm(
        input_data['interface_air_temperature'],
        input_data['interface_air_pressure'],
        input_data['gas_volume_mixing_ratios'],
        input_data['mean_air_temperature'],
        input_data['mean_air_pressure'],
        reflection='lambertian',
        num_streams=2,
        surface_emissivity=1.0,
    )
    assert np.allclose(model_output['shortwave_heating_rate'], output_data['htr_sw'])


if __name__ == '__main__':
    pytest.main([__file__])
