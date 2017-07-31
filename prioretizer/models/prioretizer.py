# encoding: utf-8

import uuid

import logging

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

    def get_scores(self, priorities, points, class_column='value', weight=True):
        """Calculate summary scores in point locations 
        
        :param priorities: Name of priorities raster (0 == low, 1 == high) 
        :param points: Vector map of locations (ground truth)
        :param class_column: Name of column to store class label (-1 == low, 1 = high) 
        :param weight:  Use weighted by class number scores if True
        :return: Value of proximity between points and raster 
        """
        rast_column = temp_name('cost', uuid.uuid4().hex)
        try:
            self.grs.grass.run_command('v.what.rast', map=points, raster=priorities, column=rast_column)
            # The simplest form of proximity: dot product
            self.grs.grass.run_command('v.db.update', map=points, column=rast_column, query_column="%s * %s" % (class_column, rast_column))
            if weight:
                positive = self.grs.grass.parse_command(
                    'v.univar', map=points,
                    column=rast_column, where="%s>0" % (class_column),
                    flags='g')
                negative = self.grs.grass.parse_command(
                    'v.univar', map=points,
                    column=rast_column, where="%s<0" % (class_column),
                    flags='g')
                result = float(positive['mean']) / float(positive['n']) + float(negative['mean']) / float(negative['n'])
            else:
                stat = self.grs.grass.parse_command('v.univar', map=points, column=rast_column, flags='g')
                result = float(stat['mean'])
        finally:
            self.grs.grass.run_command('v.db.dropcolumn', map=points, columns=rast_column)

        return result

class Optimizer:
    def __init__(
            self,
            grass,
            wood_stocks,
            road_types, wood_types,
            walking_model, wood_model,
            class_points, class_column,
            log_file=None
    ):
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

        # Logging
        self.log_file = log_file
        self.logger = logging.getLogger(self.log_file)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s', datefmt='%b %d %H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Count of loss function call
        self.counter = 0


    def optimize(self, x0, nsteps=100, nshows=5):
        x = x0
        for i in range(nshows):
            data = optimize.fmin(self._loss, x, maxiter=nsteps, full_output=True)

            x, y, iter, calls, warn = data
            print 'xy', x, y

        return x

    def _from_flatten_params(self, x):
        assert len(x) == self.param_count

        raster_costs = []
        for i, road in enumerate(self.road_types):
            raster_costs.append(RasterCost(road, x[i]))

        walk_params = WalkingCostParams(self.wood_stocks, raster_costs)

        species_costs = []
        for i, spec in enumerate(self.wood_types):
            param_number = len(self.road_types) + i * 4
            species_costs.append(
                SpecieCost(spec, x[param_number], x[param_number + 1], x[param_number + 2], x[param_number + 3])
            )

        wood_params = WoodCostParams(species_costs, x[-3], x[-2])

        alpha = x[-1]

        return (walk_params, wood_params, alpha)

    def _penalty(self, x):
        # coefs must be finite numbers => sum of the coefs must be close to 1
        s = np.sum(x[:-1])  # the last param is parameter of logistic function, it can be >1
        barier1 = (1 - s)**2

        # coefs must be >= 0
        barier2 = -1000.0 * np.sum((x < 0).astype(np.int) * x)

        return barier1 + barier2

    def _loss(self, x):
        """Loss function for optimization
        
        :param x: list of numbers. The numbers are mapped to model parameters.
        """
        # NB: format of x is strictly inherited from model. Change of the model must be followed by changen in x

        self.counter += 1

        walk_params, wood_params, alpha = self._from_flatten_params(x)

        prior = Prioretizer(grs, walk_c, wood_c)
        priorities = temp_name('prior', uuid.uuid4().hex)
        try:
            prior.calc_priorities(priorities, walk_params, wood_params, alpha)
            # We use MINIMIZER => multiply the scores by -1
            scores = -float(prior.get_scores(priorities, self.class_points, self.class_column, weight=True))
        finally:
            self.grs.grass.run_command('g.remove', type='rast', name=priorities, flags='f')

        result = scores + self._penalty(x)

        if self.log_file is not None:
            self.logger.debug('scores = %s, counter = %s, params = %s' % (scores, self.counter, x))

        return result



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

    optimizer = Optimizer(
        grs,
        'wood_stocks',
        [rc.RasterName for rc in raster_costs],
        [sp.label for sp in species_costs],
        walk_c, wood_c,
        'test', 'value',
        log_file='optimization.log'
    )

    # Flattened parameters of the model
    coefs = [
        1.37095649e-03,   2.02465935e-01,   1.81963054e-02,   2.83265685e-02,
        3.89390394e-02,   1.33573570e-02,   6.77552132e-02,   6.18582547e-02,
        7.46590691e-02,   1.41115345e-02,   8.95861468e-03,   9.96205509e-03,
        1.19912129e-01,   1.37706994e-02,   1.14446084e-02,   1.08629313e-02,
        1.03890675e-01,   2.99627408e-02,   1.51425264e-02,   8.85908783e-03,
        1.03905329e-01,   2.03990801e-02,   1.36168678e-02,   7.69956102e-03,
        6.10754265e-04,   9.48158929e-04,   8.67935842e-05
    ]

    model_params = np.array(coefs)
    result = optimizer.optimize(model_params, nsteps=2000, nshows=1)
    print 'res', result
    print 'Params', optimizer._from_flatten_params(result)

    params = optimizer._from_flatten_params(result)
    prior = Prioretizer(grs, walk_c, wood_c)
    prior.calc_priorities('tmp_prior1', params[0], params[1], alpha=params[2], overwrite=True)
    print 'Prior', prior.get_scores('tmp_prior1', 'test', 'value')

    """
    walk_params = WalkingCostParams(stocks='wood_stocks',
                                    costs_list=[RasterCost(RasterName='road_asfalt', WalkingCost=0.00135810273),
                                                RasterCost(RasterName='road_background', WalkingCost=0.200120044),
                                                RasterCost(RasterName='road_good_grunt', WalkingCost=0.0186307157),
                                                RasterCost(RasterName='road_grunt', WalkingCost=0.0282360824),
                                                RasterCost(RasterName='road_land', WalkingCost=0.0388305908),
                                                RasterCost(RasterName='road_other', WalkingCost=0.0133757199),
                                                RasterCost(RasterName='road_trop', WalkingCost=0.0669776983),
                                                RasterCost(RasterName='road_wood', WalkingCost=0.0615411286)])
    wood_params = WoodCostParams(
        woodcosts=[SpecieCost(label='DUB', wood_cost=0.0779503791, d_cost=0.0144156389, h_cost=0.00886547867,
                              b_cost=0.0100083793),
                   SpecieCost(label='LIPA', wood_cost=0.119438395, d_cost=0.0139807213, h_cost=0.0114082493,
                              b_cost=0.0110603359),
                   SpecieCost(label='KEDR', wood_cost=0.103860385, d_cost=0.0295966978, h_cost=0.0155710759,
                              b_cost=0.00876444379),
                   SpecieCost(label='JASEN', wood_cost=0.103857098, d_cost=0.0205588556, h_cost=0.0135884104,
                              b_cost=0.00764558116)], persp_factor=0.000602364714, background_cost=0.000930362805)
    alpha = 7.75671128e-05

    prior = Prioretizer(grs, walk_c, wood_c)
    prior.calc_priorities('tmp_prior', walk_params, wood_params, alpha=alpha, overwrite=True)
    print prior.get_scores('tmp_prior', 'test', 'value')

    """


