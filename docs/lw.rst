
Longwave Radiation (``LW``)
===========================

The longwave radiation component of RRTM is supported with the
following restrictions:

- No cross-sections, so ``IXSECT = 0``
- Specified temperature and pressure profiles, so ``IATM = 0``
- No clouds, so ``ICLD = 0``
- Output fluxes only for 10-3250 cm-1, so ``IOUT = 0``

These restrictions could probably be lifted relatively easily so
please contact me if you are interested in helping to support these
features.

The longwave radiation code is accessed using the ``LW`` class. The
example below (see :ref:`lw.example`) shows a simple usage of the
radiation code. The full documentation for the ``LW`` class is given
in :ref:`lw.doc`.

.. _lw.example:

Example
-------

Here is a simple example that can be found in the ``examples/basic.py``::
  
  import pyrrtm
  import numpy
  import scipy.io.netcdf
  import pylab
  
  # read in the data
  nc = scipy.io.netcdf.netcdf_file('tests_data.nc', 'r')
  tavel = nc.variables['tavel'][:] # layer average temperature
  pavel = nc.variables['pavel'][:] # layer average pressure
  tz = nc.variables['tz'][:]       # level temperature
  pz = nc.variables['pz'][:]       # level pressure
  co2 = nc.variables['co2'][:]
  o3 = nc.variables['o3'][:]
  h2o = nc.variables['h2o'][:]
  nc.close()
  
  # set up the radiation code
  nlayers = len(tavel)
  lw = pyrrtm.LW(nlayers)
  lw.tavel = tavel
  lw.pavel = pavel
  lw.tz = tz
  lw.pz = pz
  lw.set_species('co2', co2, 'molecules/cm2')
  lw.set_species('h2o', h2o, 'molecules/cm2')
  lw.set_species('o3', o3, 'molecules/cm2')
  
  # run the radiation code
  output = lw.run()
  
  # plot the result
  pylab.figure(1)
  pylab.clf()
  pylab.plot(output.htr, pavel)
  pylab.xlim(-4, 0)
  pylab.ylim(1000, 0)
  pylab.show()

The ``output`` object has the following members::

  output.totuflux   # Upward flux through each level (W/m2)
  output.totdflux   # Downward flux through each level (W/m2)
  output.fnet       # Net flux through each level (W/m2)
  output.htr        # Heating rate for each layer (K/day)
  
Each member is a numpy array.

.. _lw.doc:

The ``LW`` Class
----------------

.. class:: LW(nlayers)
   
   Performs the longwave radiative transfer calculations. ``nlayers``
   must be an integer and sets the number of layers in the model.
   
.. _lw.profile:
   
The Profile of Temperature and Pressure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   
.. attribute:: LW.tavel

   The average temperature of each layer. The layers are specified in
   ascending order.

   :units: Kelvin
   :shape: ``(nlayers, )``

.. attribute:: LW.pavel

   The average pressure of each layer. The layers are specified in
   ascending order, so ``pavel[i] > pavel[i+1]``.

   :units: hPa
   :shape: ``(nlayers, )``

.. attribute:: LW.tz

   The temperature at each level. Levels are specified in ascending order.

   :units: Kelvin
   :shape: ``(nlayers+1, )``

.. attribute:: LW.pz

   The pressure at each level. The levels are specified in ascending order.

   :units: hPa
   :shape: ``(nlayers+1, )``


.. note::

   RRTM requires both layer average and level data for pressure and
   temperature.

   Given the level data :attr:`LW.pz` and :attr:`LW.tz`, the layer
   average data :attr:`LW.pavel` and :attr:`LW.tavel` can be
   reconstructed assuming linear interpolation of temperature between
   levels. Therefore, if :attr:`LW.pz` and :attr:`LW.tz` are
   specified, :attr:`LW.pavel` and :attr:`LW.tavel` are not required
   and will be calculated automatically.

   Similarly, given :attr:`LW.pavel` and :attr:`LW.tavel`, values for
   :attr:`LW.pz` and :attr:`LW.tz` can be reconstructed. However, this
   is not a well posed problem, and so the method used is heuristic,
   and looks for smooth profiles that are consistent with the layer
   average data. In these circumstances, the resulting profiles for
   :attr:`LW.pz` and :attr:`LW.tz` should be good enough for most
   purposes, but you should check that they are satisfactory before
   using them.


.. _lw.surface:

Surface
^^^^^^^

.. attribute:: LW.tbound
   
   The surface temperature. A value of -1 means that the code will use
   ``tz[0]`` as the surface temperature.

   :units: Kelvin
   :type: ``float``

.. attribute:: LW.semis
   
   Surface emissivity. 0.0 would correspond to no long wave emission
   from the surface. Default is 1.0.


.. _lw.chemical:

Chemical Composition
^^^^^^^^^^^^^^^^^^^^

.. method:: LW.get_species(species, unit='vmr')

   Gets a profile of the concentration of one of the chemical
   species.
   
   :param species: String, one of 'H2O', 'CO2', 'O3', 'N2O', 'CO',
                   'CH4', 'O2' (case insensitive).
   :param unit: One of 'vmr', 'mmr', 'molecules/cm2'. Specifies the
                unit of ``value``.
   :returns: Array of shape ``(nlayers,)``

.. method:: LW.set_species(species, value, unit='vmr')

   Sets a profile of one of the chemical species.
   
   :param species: String, one of 'H2O', 'CO2', 'O3', 'N2O', 'CO',
                   'CH4', 'O2' (case insensitive).
   :param value: (Array of shape ``(nlayers,)`` or float). Specifies
                 the concentration of the gas in each layer. If a
                 float, the concentration is constant for all layers.
   :param unit: One of 'vmr', 'mmr', 'molecules/cm2'. Specifies the
                unit of ``value``.


.. _lw.run:

Running the Model
^^^^^^^^^^^^^^^^^

.. method:: LW.run()

   Performs the radiative transfer using RRTM.

   :returns: An instance of the :class:`Output` class.
   :raises: On an error in the RRTM code, raises an instance of :class:`RRTMError`.

.. class:: Output
   
   .. attribute:: totuflux
   
      The total upwelling flux (integrated over all bands) through each
      level.
   
      :units: W/m2
      :shape: ``(nlayers + 1, )``
   
   .. attribute:: totdflux
   
      The total downwelling flux (integrated over all bands) through each
      level.
   
      :units: W/m2
      :shape: ``(nlayers + 1, )``
   
   .. attribute:: fnet
   
      The net flux through each level. Equal to ``totdflux + totuflux``.
   
      :units: W/m2
      :shape: ``(nlayers + 1, )``
   
   .. attribute:: htr
   
      The heating rate for each layer.
   
      :units: K/day
      :shape: ``(nlayers, )``


.. _lw.adv-args:

Advanced arguments
^^^^^^^^^^^^^^^^^^

These arguments control the details of how the radiation calculation
is performed.
   
.. attribute:: LW.ireflect
   
   0 for Lambertian reflection (default).  1 for specular reflection,
   where angle is equal to downwelling angle.
               
.. attribute:: LW.iscat

   0 for no scattering. 1 for no scattering, but the calculation is
   performed using the DISORT code. 2 includes scattering but does not
   do anything as we have not allowed aerosols or clouds. Default
   is 0.

.. attribute:: LW.numangs

   0, 1, 2 or 3. Controls the number of angles used by the radiation
   scheme as quadrature points if ``iscat = 0``, or the number of
   streams if ``iscat = 1``. The default is 2.
