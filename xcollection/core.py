from __future__ import absolute_import, division, print_function

import os
import importlib
import logging

import copy

from datetime import datetime
from subprocess import check_call

import dask
import esmlab
import intake
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from tqdm import tqdm

from .config import SETTINGS, USER

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class operator(object):
    def __init__(self, function, applied_method=None, module=None, kwargs={}):
        """Initialize attributes"""

        self.applied_method = applied_method
        self.module = module
        self.function = function
        self.kwargs = kwargs

    def __repr__(self):
        """Return string represention."""
        return getattr(importlib.import_module(self.module), self.function).__repr__()

    def __call__(self, val):
        """Call the function!"""
        return getattr(importlib.import_module(self.module), self.function)(val, **self.kwargs)


class analysis(object):
    """
    A class to define and run an analysis.
    """

    def __init__(self, **kwargs):

        self.name = kwargs.pop('name')
        self.description = kwargs.pop('description', None)
        self.operators = kwargs.pop('operators', None)

    def __call__(self, dset, dsrc_applied_methods):
        """exucute sequence of operations defining an analysis.
        """
        computed_dset = dset.copy()
        applied_methods = []
        for op in self.operators:
            if op.applied_method not in dsrc_applied_methods:
                logger.info(f'applying operator: {op}')
                computed_dset = op(computed_dset)
                if op.applied_method:
                    applied_methods.append(op.applied_method)

        return computed_dset, applied_methods

    def __repr__(self):
        return repr(self.__dict__)


class analyzed_collection(object):
    """
    Run an analysis.
    """

    def __init__(
        self,
        collection_obj,
        analysis_recipe,
        analysis_name=None,
        overwrite_existing=False,
        file_format='nc',
        xr_open_kwargs=dict(decode_times=False, decode_coords=False),
        **query,
    ):

        self.collection = collection_obj
        self.query = query
        self.catalog = self.collection.search(**self.query)

        self.analysis = analysis(**analysis_recipe)
        self.cache_directory = SETTINGS['cache_directory']
        self._ds_open_kwargs = xr_open_kwargs

        self.applied_methods = []
        self.variables = None
        self.ensembles = None

        if file_format not in ['nc', 'zarr']:
            raise ValueError(f'unknown file format: {file_format}')
        self.file_format = file_format

        self._set_analysis_name(analysis_name)

        self._run_analysis(overwrite_existing=overwrite_existing)

    def _set_analysis_name(self, analysis_name):
        if not analysis_name:
            self.name = self.catalog._name + '-' + self.analysis.name
        else:
            self.name = analysis_name

    def _run_analysis(self, overwrite_existing):
        """Process data"""

        query = copy.deepcopy(self.query)
        self.ensembles = self.catalog.results.ensemble.unique()
        self.variables = self.catalog.results.variable.unique()

        self.cache_files = []
        for ens_i in self.ensembles:
            query['ensemble'] = ens_i

            cache_file = self._set_cache_file(ens_i, overwrite_existing)
            self.cache_files.append(cache_file)

            if os.path.exists(cache_file):
                continue

            catalog_subset = self.collection.search(**query)
            query_df = catalog_subset.results
            dsi = catalog_subset.to_xarray()
            # TODO: this is not implemented upstream in intake-esm
            if 'applied_methods' in query_df:
                applied_methods = query_df.applied_methods.unique()[0].split(',')
            else:
                applied_methods = []

            dso, applied_methods = self.analysis(dsi, applied_methods)
            self.applied_methods.append(applied_methods)
            # write cache file
            self._write_cache_file(cache_file, dso)


    def to_xarray(self):
        """Load the cached data."""

        ds_list = []
        for f in self.cache_files:
            ds_list.append(self._open_cache_file(f))

        return xr.concat(ds_list, dim='ens', data_vars=self.variables)

    def _open_cache_file(self, filename):
        """Open a dataset using appropriate method."""

        if self.file_format == 'nc':
            ds = xr.open_mfdataset(filename, chunks={'time': 1}, **self._ds_open_kwargs)

        elif self.file_format == 'zarr':
            ds = xr.open_zarr(filename, **self._ds_open_kwargs)

        return ds

    def _set_cache_file(self, ensemble, overwrite_existing):

        cache_file = '.'.join([self.name, '%03d' % ensemble, self.file_format])

        cache_file = os.path.join(self.cache_directory, cache_file)

        if os.path.exists(cache_file) and overwrite_existing:
            logger.info(f'removing old {cache_file}')
            check_call(['rm', '-fr', cache_file])

        return cache_file

    def _write_cache_file(self, cache_file, ds):
        """Write output to cache_file
           - add file-level attrs
           - switch method based on file extension
        """

        if os.path.exists(cache_file):
            logger.info(f'removing old {cache_file}')
            check_call(['rm', '-fr', cache_file])  # zarr files are directories
            # how to remove files and directories
            # with os package?

        time_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dsattrs = {
            'history': f'created by {USER} on {time_string}',
            'xcollection_name': self.name,
            'xcollection_analysis': repr(self.analysis),
            'xcollection_applied_methods': repr(self.applied_methods),
        }

        ds.attrs.update(dsattrs)

        logger.info(f'writing {cache_file}')
        if self.file_format == 'nc':
            ds.to_netcdf(cache_file)

        elif self.file_format == 'zarr':
            ds.to_zarr(cache_file)

        return cache_file
