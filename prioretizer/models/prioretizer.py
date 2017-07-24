# encoding: utf-8

import uuid

from walking_cost import WalkingCost, RasterCost, WalkingCostParams
from wood_cost import WoodCost, SpecieCost, WoodCostParams

from utils import temp_name

class Prioretizer:
    def __init__(self, grass, walking_cost, wood_cost):
        self.grs = grass

        self.walking_cost = walking_cost
        self.wood_cost = wood_cost

    def calc_priorities(self, output, walk_params, wood_params, overwrite=False):
        prefix = uuid.uuid4().hex
        walk_rast = temp_name('walk', prefix)
        wood_rast = temp_name('wood', prefix)

        try:
            self.wood_cost.wood_cost(wood_rast, wood_params, overwrite=overwrite)
            self.walking_cost.walking_cost(
                walking_cost=walk_rast, raster_costs_params=walk_params,
                overwrite=overwrite)

            self.grs.grass.mapcalc(
                "${out} = 1.0/(1 + exp(-(${wood} - ${walk})))",
                out=output, wood=wood_rast, walk=walk_rast,
                overwrite=overwrite)
        finally:
            self.grs.grass.run_command('g.remove', type='rast', name=wood_rast, flags='f')
            self.grs.grass.run_command('g.remove', type='rast', name=walk_rast, flags='f')

    def get_scores(self, points, priorities):
        pass


if __name__ == "__main__":
    from ..grasslib.configurator import Params
    from ..grasslib.grasslib import GRASS

    from ..models.utils import temp_name

    from ..defaults import DEFAULT_CONFIG_NAME

    config_params = Params(DEFAULT_CONFIG_NAME)

    grass_lib = config_params.grass_lib
    grass_exec = config_params.grass_exec

    location = config_params.location
    dbase = config_params.grassdata

    grs = GRASS(
        gisexec=grass_exec,
        gisbase=grass_lib,
        grassdata=dbase,
        location=location,
        init_loc=True
    )

    walk_c = WalkingCost(grs)

    raster_costs = [
        RasterCost('road_asfalt', 0.1),
        RasterCost('road_background', 1.0),
        RasterCost('road_good_grunt', 0.2),
        RasterCost('road_grunt', 0.3),
        RasterCost('road_land', 0.4),
        RasterCost('road_other', 0.5),
        RasterCost('road_trop', 0.6),
    ]
    walk_params = WalkingCostParams('wood_stocks', raster_costs)

    wood_map = 'test_woods'
    wood_c = WoodCost(
        grs, wood_map,
        forest_type_column='Mr1',
        forest_count_column='Kf1',
        diameter_column='D1',
        height_column='H1',
        perspective_column='Psp1',
        bonitet_column='bonitet',
        cumulative_cost='cost'
    )

    species_costs = [
        # 'DUB', 'LIPA', 'KEDR', 'JASEN'
        SpecieCost('DUB', 20, 3, 5, 2),
        SpecieCost('LIPA', 20, 3, 5, 2),
        SpecieCost('KEDR', 20, 3, 5, 2),
        SpecieCost('JASEN', 20, 3, 5, 2)
    ]

    wood_params = WoodCostParams(species_costs, 3, 10)

    prior = Prioretizer(grs, walk_c, wood_c)

    prior.calc_priorities('tmp_prior', walk_params, wood_params, overwrite=True)

    




