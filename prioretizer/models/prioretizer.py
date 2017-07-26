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

    def calc_priorities(self, output, walk_params, wood_params, alpha=0.01, overwrite=False):
        prefix = uuid.uuid4().hex
        walk_rast = temp_name('walk', prefix)
        wood_rast = temp_name('wood', prefix)

        try:
            self.wood_cost.wood_cost(wood_rast, wood_params, overwrite=overwrite)
            self.walking_cost.walking_cost(
                walking_cost=walk_rast, raster_costs_params=walk_params,
                overwrite=overwrite)
            self.grs.grass.mapcalc(
                "${out} = 1.0 / ( 1 + exp( -${alpha}*(${wood} - ${walk}) ) )",
                out=output, wood=wood_rast, walk=walk_rast, alpha=alpha,
                overwrite=overwrite)
        finally:
            self.grs.grass.run_command('g.remove', type='rast', name=wood_rast, flags='f')
            self.grs.grass.run_command('g.remove', type='rast', name=walk_rast, flags='f')

    def get_scores(self, priorities, points, class_column='value'):
        """Calculate summary scores in point locations 
        
        :param priorities: Name of priorities raster (0 == low, 1 == high) 
        :param points: Vector map of locations (ground truth)
        :param class_column: Name of column to store class label (0 == low, 1 = high) 
        :return: Value of proximity between points and raster 
        """
        rast_column = temp_name('cost', uuid.uuid4().hex)
        try:
            self.grs.grass.run_command('v.what.rast', map=points, raster=priorities, column=rast_column)
            # The simplest form of proximity: dot product
            self.grs.grass.run_command('v.db.update', map=points, column=rast_column, query_column="%s * %s" % (class_column, rast_column))
            stat = self.grs.grass.parse_command('v.univar', map=points, column=rast_column, flags='g')
        finally:
            self.grs.grass.run_command('v.db.dropcolumn', map=points, columns=rast_column)

        return stat['mean']



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
        SpecieCost('DUB', 15.0, 20, 15, 1),
        SpecieCost('LIPA', 12.0, 20, 15, 1),
        SpecieCost('KEDR', 14.0, 20, 15, 1),
        SpecieCost('JASEN', 13.0, 20, 15, 1)
    ]

    wood_params = WoodCostParams(species_costs, 1, 100)

    prior = Prioretizer(grs, walk_c, wood_c)

    prior.calc_priorities('tmp_prior', walk_params, wood_params, overwrite=True)

    print prior.get_scores('tmp_prior', 'test', 'value')






