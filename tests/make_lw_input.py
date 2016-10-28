import scipy.io.netcdf
import pickle

def make_input_file(filename, iscat, numangs,
                    tbound, ireflect, semis,
                    tavel, pavel, tz, pz, wkl, wbrodl):
    nmol, nlayers = wkl.shape

    # some sanity checks.
    len(tavel) == len(pavel) == len(wbrodl) == nlayers
    len(tz) == len(pz) == nlayers + 1
    
    # now make the file
    nc = scipy.io.netcdf.netcdf_file(filename, 'w')

    # attributes required
    nc.lw_iscat = iscat
    nc.lw_numangs = numangs
    nc.lw_tbound = tbound
    nc.lw_ireflect = ireflect
    nc.lw_semis = semis
    
    # now fill up the variables.
    nc.createDimension("layer", nlayers)
    nc.createDimension("level", nlayers + 1)
    nc.createDimension("mol", nmol)
    nc.createVariable('pavel', 'd', ('layer',))[:] = pavel
    nc.createVariable('tavel', 'd', ('layer',))[:] = tavel
    nc.createVariable('pz', 'd', ('level',))[:] = pz
    nc.createVariable('tz', 'd', ('level',))[:] = tz
    nc.createVariable('wbrodl', 'd', ('layer',))[:] = wbrodl
    nc.createVariable('wkl', 'd', ('layer', 'mol'))[:] = wkl.T

    nc.close()


inputs = pickle.load(open("inputs.dat", 'r'))
inputs['semis'] = 1
inputs['iscat'] = 0
inputs['numangs'] = 2
make_input_file('input_mls.nc', **inputs)
inputs['iscat'] = 1
make_input_file('input_mls_disort.nc', **inputs)
inputs['iscat'] = 2
make_input_file('input_mls_disort2.nc', **inputs)
