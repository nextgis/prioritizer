# encoding: utf-8

import uuid

from scipy import optimize
import numpy as np

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
                out=output, wood=wood_rast, walk=walk_rast, alpha=format(alpha, '.10f'),
                overwrite=overwrite)
        finally:
            self.grs.grass.run_command('g.remove', type='rast', name=wood_rast, flags='f')
            self.grs.grass.run_command('g.remove', type='rast', name=walk_rast, flags='f')

    def get_scores(self, priorities, points, class_column='value'):
        """Calculate summary scores in point locations 
        
        :param priorities: Name of priorities raster (0 == low, 1 == high) 
        :param points: Vector map of locations (ground truth)
        :param class_column: Name of column to store class label (-1 == low, 1 = high) 
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

class Optimizer:
    def __init__(self, grass, wood_stocks, road_types, wood_types, walking_model, wood_model, class_points, class_column):
        self.grs = grass

        self.wood_stocks = wood_stocks
        self.road_types = road_types
        self.wood_types = wood_types

        self.walking_model = walking_model
        self.wood_model = wood_model

        self.class_points = class_points
        self.class_column = class_column

        #                  road_costs,         (wood spec params==4), persp_fact, background, alpha
        self.param_count = len(self.road_types) + len(self.wood_types) * 4 + 2 + 1

    def optimize(self, x0, nsteps=10, nshows=50):
        x = x0
        for i in range(nshows):
            x, y, iter, funcals, _, sols = optimize.fmin(self._loss, x, maxiter=nsteps)
            print x, y, iter, funcals

        return x

    def _barrier(self, x):
        # coefs must be finite numbers => sum of the coefs must be close to 1
        s = np.sum(x)
        barier1 = (1 - s)**2

        # coefs must be >= 0
        barier2 = -1000.0 * np.sum((x < 0).astype(np.int) * x)

        return barier1 + barier2

    def _loss(self, x):
        """Loss function for optimization
        
        :param x: list of numbers. The numbers are mapped to model parameters.
        """
        # NB: format of x is strictly inherited from model. Change of the model must be followed by changen in x

        assert len(x) == self.param_count

        raster_costs = []
        for i, road in enumerate(self.road_types):
            raster_costs.append(RasterCost(road, x[i]))

        walk_params = WalkingCostParams(self.wood_stocks, raster_costs)

        species_costs = []
        for i, spec in enumerate(self.wood_types):
            param_number = len(self.road_types) + i*4
            species_costs.append(
                SpecieCost(spec, x[param_number], x[param_number+1], x[param_number+2], x[param_number+3])
            )

        wood_params = WoodCostParams(species_costs, x[-3], x[-2])

        prior = Prioretizer(grs, walk_c, wood_c)
        priorities = temp_name('prior', uuid.uuid4().hex)
        try:
            prior.calc_priorities(priorities, walk_params, wood_params, alpha=x[-1], overwrite=True)
            scores = -float(prior.get_scores(priorities, self.class_points, self.class_column))
        finally:
            self.grs.grass.run_command('g.remove', type='rast', name=priorities, flags='f')

        return scores + self._barrier(x)



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

    raster_costs = [
        RasterCost('road_asfalt', 0.01),
        RasterCost('road_background', 0.73),
        RasterCost('road_good_grunt', 0.02),
        RasterCost('road_grunt', 0.03),
        RasterCost('road_land', 0.04),
        RasterCost('road_other', 0.05),
        RasterCost('road_trop', 0.06),
        RasterCost('road_wood', 0.06),
    ]
    # walk_params = WalkingCostParams('wood_stocks', raster_costs)
    species_costs = [
        SpecieCost('DUB', 15.0, 20, 15, 1),
        SpecieCost('LIPA', 12.0, 20, 15, 1),
        SpecieCost('KEDR', 14.0, 20, 15, 1),
        SpecieCost('JASEN', 13.0, 20, 15, 1)
    ]

    # wood_params = WoodCostParams(species_costs, 1, 100)

    # prior = Prioretizer(grs, walk_c, wood_c)
    # prior.calc_priorities('tmp_prior', walk_params, wood_params, overwrite=True)
    # print prior.get_scores('tmp_prior', 'test', 'value')

    optimizer = Optimizer(
        grs,
        'wood_stocks',
        [rc.RasterName for rc in raster_costs],
        [sp.label for sp in species_costs],
        walk_c, wood_c,
        'test', 'value'
    )

    # Flattened parameters of the model
    model_params = np.array([
        0.001, 0.23, 0.02, 0.03, 0.04, 0.05, 0.06, 0.06,
        0.1, 0.020, 0.015, 0.01,
        0.1, 0.020, 0.015, 0.01,
        0.1, 0.020, 0.015, 0.01,
        0.1, 0.020, 0.015, 0.01,
        0.001, 0.001,
        0.00001
    ])
    print optimizer.optimize(model_params)






