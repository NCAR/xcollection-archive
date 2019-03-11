import esmlab
import intake
import intake_esm

intake_esm.config.set(dict(database_directory='tests/data'))

col = intake.open_esm_metadatastore(collection_input_file='tests/data/collection_input.yml')
