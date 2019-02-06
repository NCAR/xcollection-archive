from __future__ import absolute_import, division, print_function

import os
import importlib
import logging

from datetime import datetime
from subprocess import check_call
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import esmlab
import intake

from .config import SETTINGS, USER

logging.basicConfig(level=logging.DEBUG)


class yaml_operator(yaml.YAMLObject):
    """A wrapper used for defining callable functions in YAML.

    For example:
    !operator
    applied_methods: ['time:clim_mon']
    module: esmlab.climatology
    function: compute_mon_climatology
    kwargs: {}
    """

    yaml_tag = "!operator"

    def __init__(self, applied_method, module, function, kwargs):
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
        return getattr(importlib.import_module(self.module), self.function)(
            val, **self.kwargs
        )


class analysis(object):
    """
    A class to define and run an analysis.
    """

    def __init__(self, **kwargs):

        self.name = kwargs.pop("name")
        self.description = kwargs.pop("description", None)
        self.operators = kwargs.pop("operators", None)
        self.sel_kwargs = kwargs.pop("sel_kwargs", None)
        self.isel_kwargs = kwargs.pop("isel_kwargs", None)

    def __call__(self, dset, dsrc_applied_methods):
        computed_dset = dset.copy()

        if self.sel_kwargs:
            logging.info(f"applying sel_kwargs: {self.sel_kwargs}")
            computed_dset = computed_dset.sel(**self.sel_kwargs)

        if self.isel_kwargs:
            logging.info(f"applying isel_kwargs: {self.isel_kwargs}")
            computed_dset = computed_dset.isel(**self.isel_kwargs)

        applied_methods = []
        for op in self.operators:
            if op.applied_method not in dsrc_applied_methods:
                logging.info(f"applying operator: {op}")
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
        collection,
        analysis_recipe,
        analysis_name=None,
        overwrite_existing=False,
        file_format="nc",
        **query,
    ):

        col_obj = intake.open_cesm_metadatastore(collection)
        self.catalog = col_obj.search(**query)
        self.analysis = analysis(**analysis_recipe)
        self.applied_methods = []
        self.cache_directory = SETTINGS["cache_directory"]

        if file_format not in ["nc", "zarr"]:
            raise ValueError(f"unknown file format: {file_format}")
        self.file_format = file_format

        self._set_analysis_name(analysis_name)

        self._run_analysis(overwrite_existing=overwrite_existing)

    def _set_analysis_name(self, analysis_name):
        if not analysis_name:
            self.name = self.catalog._name + "-" + self.analysis.name
        else:
            self.name = analysis_name

    def _run_analysis(self, overwrite_existing):
        """Process data"""

        query = dict(self.catalog.query)
        ensembles = self.catalog.results.ensemble.unique()
        variables = self.catalog.results.variable.unique()

        self.cache_files = []
        for ens_i in ensembles:
            query["ensemble"] = ens_i

            cache_file = self._set_cache_file(ens_i, overwrite_existing)
            self.cache_files.append(cache_file)

            if os.path.exists(cache_file):
                continue

            dsi = xr.Dataset()
            for var_i in variables:
                query["variable"] = var_i

                query_results = self._get_subset(query)

                files = query_results.files.tolist()
                year_offset = query_results.year_offset.unique()[0]

                # TODO: this is not implemented upstream in xcollection
                if "applied_methods" in query_results:
                    applied_methods = query_results.applied_methods.unique()[0].split(',')
                else:
                    applied_methods = []

                dsi = xr.merge(
                    (
                        dsi,
                        xr.open_mfdataset(
                            files,
                            decode_times=False,
                            decode_coords=False,
                            data_vars=[var_i],
                            chunks={"time": 1},
                        ),
                    )
                )

            # apply the analysis
            dso = self._fixtime(dsi, year_offset)
            dso, applied_methods = self.analysis(dso, applied_methods)
            self.applied_methods.append(applied_methods)
            dso = self._unfixtime(dso)

            # write cache file
            self._write_cache_file(cache_file, dso)

    def _get_subset(self, query):
        """ Get a subset of collection entries that match a query """
        df = self.catalog.results

        condition = np.ones(len(df), dtype=bool)

        for key, val in query.items():

            if isinstance(val, list):
                condition_i = np.zeros(len(df), dtype=bool)
                for val_i in val:
                    condition_i = condition_i | (df[key] == val_i)
                condition = condition & condition_i

            elif val is not None:
                condition = condition & (df[key] == val)

        query_results = df.loc[condition].sort_values(by=["sequence_order", "files"], ascending=True)

        return query_results

    def to_xarray(self):
        """Load the cached data."""

        ds_list = []
        for f in self.cache_files:
            ds_list.append(self._open_cache_file(f))

        return xr.concat(ds_list, dim="ens")  # , data_vars=[self.variable])

    def _open_cache_file(self, filename):
        """Open a dataset using appropriate method."""

        if self.file_format == "nc":
            ds = xr.open_mfdataset(filename, decode_coords=False, chunks={"time": 1})

        elif self.file_format == "zarr":
            ds = xr.open_zarr(filename, decode_coords=False)

        return ds

    def _set_cache_file(self, ensemble, overwrite_existing):

        cache_file = ".".join(
            [
                self.name,
                "%03d" % ensemble,
                self.file_format,
            ]
        )

        cache_file = os.path.join(self.cache_directory, cache_file)

        if os.path.exists(cache_file) and overwrite_existing:
            logging.info(f"removing old {cache_file}")
            check_call(["rm", "-fr", cache_file])

        return cache_file

    def _write_cache_file(self, cache_file, ds):
        """Write output to cache_file
           - add file-level attrs
           - switch method based on file extension
        """

        if os.path.exists(cache_file):
            logging.info(f"removing old {cache_file}")
            check_call(["rm", "-fr", cache_file])  # zarr files are directories
            # how to remove files and directories
            # with os package?

        time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dsattrs = {
            "history": f"created by {USER} on {time_string}",
            "xcollection_name": self.name,
            "xcollection_analysis_name": self.analysis_name,
            "xcollection_analysis": repr(self.analysis),
            "xcollection_applied_methods": repr(self.applied_methods)
        }

        ds.attrs.update(dsattrs)

        logging.info(f"writing {cache_file}")
        if self.file_format == "nc":
            ds.to_netcdf(cache_file)

        elif self.file_format == "zarr":
            ds.to_zarr(cache_file)

        return cache_file

    def _fixtime(self, dsi, year_offset):
        tb_name, tb_dim = esmlab.utils.time_bound_var(dsi)
        if tb_name and tb_dim:
            return esmlab.utils.compute_time_var(
                dsi, tb_name, tb_dim, year_offset=year_offset
            )
        else:
            return dsi

    def _unfixtime(self, dsi):
        return esmlab.utils.uncompute_time_var(dsi)
