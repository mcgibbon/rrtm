

[RRTM](http://rtweb.aer.com/rrtm_frame.html) is a radiation code that is widely used in atmospheric science. The code has a rather nasty ascii interface that is a big obstacle to using the code in scripts, and also inhibits performance if lots of radiative transfer calculations need doing. pyrrtm aims to address these issues, allowing RRTM to work well alongside python.

- Documentation and installation instructions are available [here](http://pyrrtm.flannaghan.com).
- Please [contact me](mailto:tomflannaghan@gmail.com) if you wish to contribute or have any questions.

Also included in pyrrtm is a stand-alone netcdf interface. This would be of use to anyone not interested in using RRTM with python (e.g. calling RRTM from another scripting language). See [`cli_doc.org`](https://github.com/tomflannaghan/pyrrtm/blob/master/cli_doc.org) for documentation on just the command line interface.
