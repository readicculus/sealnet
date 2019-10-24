def transform_removed(data):
    return data[~data.status.str.contains("removed")]

def transform_remove_bad_res(data):
    return data[~data.status.str.contains("bad_res")]

def transform_remove_off_edge(data):
    return data[~data.status.str.contains("off_edge")]

def transform_updated(data):
    return data[data.updated == True]

def transform_seal_only(data):
   return data[data.species_id.str.contains("Seal")]


def transform_seal_and_pb_only(data):
   return data[data.species_id.str.contains("Seal") | data.species_id.str.contains("Polar Bear")]

def transform_remove_unk_seal(data):
    return data[~data.species_id.str.contains("UNK Seal")]


def transform_remove_shadow_annotations(data):
    return data[~data.hotspot_id.str.endswith("s")]