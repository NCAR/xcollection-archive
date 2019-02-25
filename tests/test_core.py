import yaml
import xarray as xr

from esmlab.climatology import compute_mon_climatology
from xcollection import analyzed_collection, operator


def test_analyzed_collection():

    recipe = {'name': 'monclim',
              'description': 'compute monthly climatology',
              'operators': [
                  operator(applied_method="time:mon_clim",
                           module='esmlab.climatology',
                           function='compute_mon_climatology'),
                  ]}

    query = dict(case='cesm_pop_sample', variable='NO3')

    dc = analyzed_collection(collection='test_collection',
                             analysis_recipe=recipe,
                             overwrite_existing=True,
                             file_format="nc",
                             **query)
    assert isinstance(dc.to_xarray(), xr.Dataset)

def test_read_recipe():
    with open('tests/data/recipes.yml') as fid:
        recipes = yaml.load(fid)
    keys = list(recipes.keys())
    op0 = recipes[keys[0]]['operators'][0]
    assert isinstance(op0, operator)
