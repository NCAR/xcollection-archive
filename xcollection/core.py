from __future__ import absolute_import, division, print_function

import os
from subprocess import check_call
import yaml
import importlib
from datetime import datetime

import numpy as np
import xarray as xr
import pandas as pd

import logging

import esmlab

logging.basicConfig(level=logging.INFO)


# config
USER = os.environ['USER']
dirout = f'/glade/scratch/{USER}/calcs/processed_collections'
if not os.path.exists(dirout):
    os.makedirs(dirout)


class yaml_operator(yaml.YAMLObject):
    '''A wrapper used for defining callable functions in YAML.

    For example:
    !operator
    applied_methods: ['time:clim_mon']
    module: esmlab.climatology
    function: compute_mon_climatology
    kwargs: {}
    '''

    yaml_tag = u'!operator'

    def __init__(self, applied_method, module, function, kwargs):
        '''Initialize attributes'''
        self.applied_method = applied_method
        self.module = module
        self.function = function
        self.kwargs = kwargs

    def __repr__(self):
        '''Return string represention.'''
        return getattr(importlib.import_module(self.module),
                       self.function).__repr__()

    def __call__(self, val):
        '''Call the function!'''
        return getattr(importlib.import_module(self.module),
                       self.function)(val, **self.kwargs)


class datasource(object):
    '''
    An object describing a datasource for incorporation in a analysis.

    Attributes
    ----------
    name : string
           Name of the object

    variables : array-like
                List of variables

    ensembles : array-like
                List of ensembles in datasource, defaults to `[0]`

    applied_methods : array-like
                     List of methods applied to dataset

    year_offset : int
                  Integer year offset to align calendar
                  (i.e. adjusted_time = time + year_offset)

    '''
    def __init__(self, name, data_descriptor):
        '''
        Instantiate datasource object.

        Parameters
        ----------
        name : string
               The name of this datasource

        data_descriptor : dict or pandas.DataFrame
                          A dictionary where the keys are data attributes
                          and each entry is a list; can also be
                          a pandas.DataFrame.
        '''


        if isinstance(data_descriptor, dict):
            data_descriptor = pd.DataFrame(data_descriptor)

        self.name = name
        self.variables = data_descriptor.variable.unique()

        if 'ensemble' in data_descriptor:
            self.ensembles = data_descriptor.ensemble.unique()
        else:
            self.ensembles = [0]

        if 'applied_methods' in data_descriptor:
            self.applied_methods = data_descriptor.applied_methods.unique()
        else:
            self.applied_methods = []

        if 'year_offset' in data_descriptor:
            self.year_offset = data_descriptor.year_offset.unique()[0]
        else:
            self.year_offset = np.nan

        # generate file groups
        self.files = dict()
        self.attrs = dict()
        for ens_i in self.ensembles:
            self.files[ens_i] = dict()
            self.attrs[ens_i] = dict()
            for var_i in self.variables:
                query = (data_descriptor.variable == var_i)
                if 'ensemble' in data_descriptor:
                    query = query & (data_descriptor.ensemble == ens_i)

                data_loc = data_descriptor.loc[query]
                files = data_loc.files.tolist() # relying on ordered list in data passed
                if not files:
                    raise ValueError(f'no files for ensemble={ens_i}, variable={var_i}')
                self.files[ens_i][var_i] = files
                self.attrs[ens_i][var_i] = {key: data_loc[key].tolist()
                                             for key in data_loc
                                             if key not in ['files', 'variable', 'ensemble']}

    def __repr__(self):
        return repr(self.__dict__)

class analysis(object):
    '''
    A class to define and run an analysis.
    '''
    def __init__(self, **kwargs):

        self.description = kwargs.pop('description', None)
        self.operators = kwargs.pop('operators', None)
        self.sel_kwargs = kwargs.pop('sel_kwargs', None)
        self.isel_kwargs = kwargs.pop('isel_kwargs', None)

    def __call__(self, dset, dsrc_applied_methods):
        computed_dset = dset.copy()

        if self.sel_kwargs:
            logging.info(f'applying sel_kwargs: {self.sel_kwargs}')
            computed_dset = computed_dset.sel(**self.sel_kwargs)

        if self.isel_kwargs:
            logging.info(f'applying isel_kwargs: {self.isel_kwargs}')
            computed_dset = computed_dset.isel(**self.isel_kwargs)

        applied_methods = []
        for op in self.operators:
            if op.applied_method not in dsrc_applied_methods:
                logging.info(f'applying operator: {op}')
                computed_dset = op(computed_dset)
                if op.applied_method:
                    applied_methods.append(op.applied_method)

        return computed_dset, applied_methods

    def __repr__(self):
        return repr(self.__dict__)

class analyzed_datasource(object):
    '''
    Run an analysis.
    '''

    def __init__(self, analysis_name, analysis_recipe, datasource,
                 clobber_cache=False, file_format='nc'):

        self.analysis_name = analysis_name
        self.analysis = analysis(**analysis_recipe)
        self.datasource = datasource
        self.applied_methods = []

        self.file_format = file_format
        if self.file_format not in ['nc', 'zarr']:
            raise ValueError(f'unknown file format: {file_format}')

        self._run_analysis(clobber_cache=clobber_cache)

    def _run_analysis(self, clobber_cache):
        '''Process data'''

        self.cache_files = []
        for ens_i in self.datasource.ensembles:

            cache_file = self._set_cache_file(ens_i, clobber_cache)
            self.cache_files.append(cache_file)

            if os.path.exists(cache_file):
                continue

            dsi = xr.Dataset()
            for var_i in self.datasource.variables:
                files = self.datasource.files[ens_i][var_i]
                attrs = self.datasource.attrs[ens_i][var_i]
                year_offset = self.datasource.year_offset

                dsi = xr.merge((dsi,xr.open_mfdataset(files,
                                                      decode_times=False,
                                                      decode_coords=False,
                                                      data_vars=[var_i],
                                                      chunks={'time':1})))

            # apply the analysis
            dso = self._fixtime(dsi, year_offset)
            dso, applied_methods = self.analysis(dso, self.datasource.applied_methods)
            self.applied_methods.append(applied_methods)
            dso = self._unfixtime(dso)

            # write cache file
            self._write_cache_file(cache_file, dso)

    def to_xarray(self):
        '''Load the cached data.'''

        ds_list = []
        for f in self.cache_files:
            ds_list.append(self._open_cache_file(f))

        return xr.concat(ds_list, dim='ens') #, data_vars=[self.variable])

    def _open_cache_file(self, filename):
        '''Open a dataset using appropriate method.'''

        if self.file_format == 'nc':
            ds = xr.open_mfdataset(filename, decode_coords=False,
                                   chunks={'time':1})

        elif self.file_format == 'zarr':
            ds = xr.open_zarr(filename, decode_coords=False)

        return ds

    def _set_cache_file(self, ensemble, clobber_cache):

        cache_file = '.'.join([self.datasource.name,
                             '%03d'%ensemble,
                             self.analysis_name,
                             self.file_format])

        cache_file = os.path.join(dirout, cache_file)

        if os.path.exists(cache_file) and clobber_cache:
            logging.info(f'removing old {cache_file}')
            check_call(['rm', '-fr', cache_file])

        return cache_file

    def _write_cache_file(self, cache_file, ds):
        '''Write output to cache_file
           - add file-level attrs
           - switch method based on file extension
        '''

        if not os.path.exists(dirout):
            logging.info(f'creating {dirout}')
            os.makedirs(dirout)

        if os.path.exists(cache_file):
            logging.info(f'removing old {cache_file}')
            check_call(['rm','-fr',cache_file]) # zarr files are directories
                                              # how to remove files and directories
                                              # with os package?

        time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dsattrs = {'history': f'created by {USER} on {time_string}',
                   'analysis_name': self.analysis_name,
                   'analysis': repr(self.analysis),
                   'applied_methods': repr(self.applied_methods),
                   'datasource': repr(self.datasource)}

        ds.attrs.update(dsattrs)

        logging.info(f'writing {cache_file}')
        if self.file_format == 'nc':
            ds.to_netcdf(cache_file)

        elif self.file_format  == 'zarr':
            ds.to_zarr(cache_file)

        return cache_file

    def _fixtime(self, dsi, year_offset):
        tb_name, tb_dim = esmlab.utils.time_bound_var(dsi)
        if tb_name and tb_dim:
            return esmlab.utils.compute_time_var(dsi, tb_name, tb_dim,
                                                 year_offset=year_offset)
        else:
            return dsi

    def _unfixtime(self, dsi):
        tb_name, tb_dim = esmlab.utils.time_bound_var(dsi)
        if tb_name and tb_dim:
            return esmlab.utils.uncompute_time_var(dsi, tb_name, tb_dim)
        else:
            return dsi
