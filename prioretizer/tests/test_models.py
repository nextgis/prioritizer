#!/bin/env python
# -*- coding: utf-8 -*-


import sys
import os

import shutil
import uuid

import unittest

from ..models.model_config import ModelParams
from ..models.walking_cost import WalkingCostParams, RasterCost
from ..models.wood_cost import WoodCostParams, SpecieCost
from ..models.prioretizer import Optimizer


MODEL_CONFIG_FILE = 'prioretizer/tests/test_model.conf'

class TestModels(unittest.TestCase):
    def test_init_model_config(self):
        model = ModelParams(MODEL_CONFIG_FILE)
        model.load()

        walk = WalkingCostParams(stocks='wood_stocks',
                          costs_list=[RasterCost(RasterName='roads_asfalt', WalkingCost=0.012896805974870763),
                                      RasterCost(RasterName='roads_good_grunt', WalkingCost=0.0071922850315475819),
                                      RasterCost(RasterName='roads_bad_type', WalkingCost=3.8396939625971893e-08),
                                      RasterCost(RasterName='roads_background', WalkingCost=0.04709689266339892)])
        self.assertEqual(walk, model.walking_cost_params)

        wood = WoodCostParams(
            woodcosts=[SpecieCost(label='DUB', wood_cost=1.6255207525356973e-07),
                       SpecieCost(label='LIPA', wood_cost=0.11964072153354283),
                       SpecieCost(label='KEDR', wood_cost=0.81122661608560187),
                       SpecieCost(label='JASEN', wood_cost=4.6377110636275511e-06)],
            persp_factor=0.00045597720826173781,
            background_cost=0.004256807171225948)

        self.assertEqual(wood, model.wood_cost_params)

    def test_setters(self):
        model = ModelParams(MODEL_CONFIG_FILE)
        model.load()
        walk = WalkingCostParams(stocks='S',
                                 costs_list=[RasterCost(RasterName='A', WalkingCost=1),
                                             RasterCost(RasterName='B', WalkingCost=2),
                                             RasterCost(RasterName='C', WalkingCost=3),
                                             RasterCost(RasterName='D', WalkingCost=4)])
        model.walking_cost_params = walk

        self.assertEqual(model.stocks, walk.stocks)
        self.assertEqual(model._road_costs, walk.costs_list)
        self.assertEqual(model.road_labels, ['A', 'B', 'C', 'D'])

        wood = WoodCostParams(
            woodcosts=[SpecieCost(label='X', wood_cost=5),
                       SpecieCost(label='Y', wood_cost=6),
                       SpecieCost(label='Z', wood_cost=7)],
            persp_factor=9,
            background_cost=10)
        model.wood_cost_params =wood
        self.assertEqual(model.wood_perspect_factor, wood.persp_factor)
        self.assertEqual(model._wood_background_cost, wood.background_cost)
        self.assertEqual(model.wood_labels, ['X', 'Y', 'Z'])
        self.assertEqual(model._specie_costs, wood.woodcosts)

    def test_save_model(self):
        model = ModelParams(MODEL_CONFIG_FILE)
        model.load()

        walk = WalkingCostParams(stocks='wood_stocks',
                                 costs_list=[RasterCost(RasterName='roads_asfalt', WalkingCost=1),
                                             RasterCost(RasterName='roads_good_grunt', WalkingCost=2),
                                             RasterCost(RasterName='roads_bad_type', WalkingCost=3),
                                             RasterCost(RasterName='roads_background', WalkingCost=4)])
        wood = WoodCostParams(
            woodcosts=[SpecieCost(label='DUB', wood_cost=5),
                       SpecieCost(label='LIPA', wood_cost=6),
                       SpecieCost(label='KEDR', wood_cost=7),
                       SpecieCost(label='JASEN', wood_cost=8)],
            persp_factor=9,
            background_cost=10)
        alpha = 11

        model.update_params(walk, wood, alpha)

        temp = 'test_config'
        try:
            model.save(temp)
            model1 = ModelParams(temp)
            model1.load()
        finally:
            os.unlink(temp)

        self.assertEqual(model.wood_cost_params, model1.wood_cost_params)
        self.assertEqual(model._road_costs, model1._road_costs)
        self.assertEqual(model.func_alpha, model1.func_alpha)
        self.assertEqual(model.wood_name, model1.wood_name)
        self.assertEqual(model.wood_perspect_column, model1.wood_perspect_column)
        self.assertEqual(model.wood_perspect_factor, model1.wood_perspect_factor)
        self.assertEqual(model._wood_background_cost, model1._wood_background_cost)
        self.assertEqual(model.wood_forest_type_column, model1.wood_forest_type_column)
        self.assertEqual(model.stocks, model1.stocks)

    def test_optim_params(self):
        walk = WalkingCostParams(stocks='wood_stocks',
                          costs_list=[RasterCost(RasterName='roads_asfalt', WalkingCost=1),
                                      RasterCost(RasterName='roads_good_grunt', WalkingCost=2),
                                      RasterCost(RasterName='roads_bad_type', WalkingCost=3),
                                      RasterCost(RasterName='roads_background', WalkingCost=4)])
        wood = WoodCostParams(
            woodcosts=[SpecieCost(label='DUB', wood_cost=5),
                       SpecieCost(label='LIPA', wood_cost=6),
                       SpecieCost(label='KEDR', wood_cost=7),
                       SpecieCost(label='JASEN', wood_cost=8)],
            persp_factor=9,
            background_cost=10)
        alpha = 11

        road_types = ['roads_asfalt', 'roads_good_grunt', 'roads_bad_type', 'roads_background']
        wood_types = ['DUB','LIPA','KEDR','JASEN']
        opt = Optimizer(
            'dummy',
            walk.stocks,
            road_types, wood_types,
            'dummy', 'dummy',
            'dummy', 'dummy',
            log_file='dummy'
        )
        params = opt.flatten_params(walk, wood, alpha)
        new_params = opt._from_flatten_params(params)

        self.assertEqual(new_params[0], walk)
        self.assertEqual(new_params[1], wood)
        self.assertEqual(new_params[2], alpha)



if __name__ == '__main__':
    suite = unittest.makeSuite(TestModels, 'test')
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
