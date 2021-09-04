from utils import read_binary_grid3d
from os.path import join
data, bbox = read_binary_grid3d(join("data","smoke_mitsuba.vol"))
exit(0)