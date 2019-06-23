def transform_removed(data):
    return data[~data.status.str.contains("removed")]

def transform_updated(data):
    return data[data.updated == True]

def transform_seal_only(data):
   return data[data.species_id.str.contains("Seal")]


def transform_remove_unk_seal(data):
    return data[~data.species_id.str.contains("UNK Seal")]


