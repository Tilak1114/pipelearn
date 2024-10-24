from mbodied.types.sample import Sample
from embdata.sense.world import World

class DataSourceData(Sample):
    world: World
    uid: str

