# encoding: utf-8

import uuid

import logging

from scipy import optimize
import numpy as np

from walking_cost import WalkingCost, RasterCost, WalkingCostParams
from wood_cost import WoodCost, SpecieCost, WoodCostParams

from utils import temp_name

BIG_NUMBER = 1048576.0   # = 1024**2


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
        :return: positive value is good consistency of points and priorities raster
                 negative value is discrepancy of points and priorities raster
        """
        rast_column = temp_name('cost', uuid.uuid4().hex)
        try:
            self.grs.grass.run_command('v.db.addcolumn', map=points, columns="%s double" % (rast_column, ))
            self.grs.grass.run_command('v.what.rast', map=points, raster=priorities, column=rast_column, quiet=True)
            # The simplest form of proximity: dot product
            # NB: priorities raster contains values in (0, 1), so zeros don't influence the scores =>
            #    rescale priorities to (-1, 1).
            self.grs.grass.run_command('v.db.update', map=points, column=rast_column, query_column="%s * (2 * %s - 1)" % (class_column, rast_column))
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
        self.param_count = len(self.road_types) + len(self.wood_types) + 2 + 1

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
        x = self.flatten_params(x0[0], x0[1], x0[2])
        for i in range(nshows):
            x = optimize.fmin(self._loss, x, maxiter=nsteps, ftol=1e-6)

        return self._from_flatten_params(x)

    def _from_flatten_params(self, x):
        assert len(x) == self.param_count

        raster_costs = []
        for i, road in enumerate(self.road_types):

            # Cost can became <0 during optimization. Fix it (add penalty later)
            r_cost = x[i] if x[i] >=0 else 0
            if x[i] < 0:
               self.logger.debug('Parameter %s is changed to %s' % (x[i], r_cost))

            raster_costs.append(RasterCost(road, r_cost))

        walk_params = WalkingCostParams(self.wood_stocks, raster_costs)

        species_costs = []
        for i, spec in enumerate(self.wood_types):
            # For complex model:
            # param_number = len(self.road_types) + i * 4
            # species_costs.append(
                # SpecieCost(spec, x[param_number], x[param_number + 1], x[param_number + 2], x[param_number + 3])
            # )

            param_number = len(self.road_types) + i * 1
            species_costs.append(
                SpecieCost(spec, x[param_number])
            )

        wood_params = WoodCostParams(species_costs, x[-3], x[-2])

        alpha = x[-1]

        return (walk_params, wood_params, alpha)

    def flatten_params(self, walking_cost_params, wood_cost_params, alpha):
        road_costs = [w.WalkingCost for w in walking_cost_params.costs_list]
        species_costs = [s.wood_cost for s in wood_cost_params.woodcosts]
        persp = wood_cost_params.persp_factor
        bg_cost = wood_cost_params.background_cost

        return road_costs + species_costs + [persp, bg_cost, alpha]

    def _penalty(self, x):
        # coefs must be finite numbers => sum of the coefs must be close to 1; it allows to avoid infinite coefs
        s = np.sum(x[:-1])  # the last param is parameter of logistic function, it can be >1
        barier1 = (1 - s)**2

        # coefs must be >= 0
        barier2 = -BIG_NUMBER * np.sum((x < 0).astype(np.int) * x)

        return barier1 + barier2

    def _loss(self, x):
        """Loss function for optimization
        
        :param x: list of numbers. The numbers are mapped to model parameters.
        """
        # NB: format of x is strictly inherited from model. Change of the model must be followed by changen in x

        # self.logger.debug('Calc loss function. X = %s' % (x, ))
        self.counter += 1

        walk_params, wood_params, alpha = self._from_flatten_params(x)

        prior = Prioretizer(self.grs, self.walking_model, self.wood_model)
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
