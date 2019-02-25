import intake
import intake_esm

import esmlab

intake_esm.set_options(database_directory='tests/data')
build_kwargs = dict(collection_input_file='tests/data/collection_input.yml',
                    collection_type_def_file='tests/data/cesm_definitions.yml',
                    overwrite_existing=True,
                    include_cache_dir=False)

col = intake.open_cesm_metadatastore('test_collection', build_args=build_kwargs)
