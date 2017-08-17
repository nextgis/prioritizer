# encoding: utf-8

from ConfigParser import ConfigParser

from wood_cost import SpecieCost, WoodCostParams, WoodCost
from walking_cost import WalkingCostParams, WalkingCost, RasterCost

class ConfigError(Exception):
    pass

class ModelParams:
    def __init__(self, config_file):
        self.config = ConfigParser()
        self.config.readfp(open(config_file))

        self.wood_name = self.config.get('WOOD', 'name')
        # Запас основной породы
        self.wood_perspect_column = self.config.get('WOOD', 'perspect_column')
        # Относительная значимость запаса
        self.wood_perspect_factor = float(self.config.get('WOOD', 'perspect_factor'))

        # Относительная стоимость древесины вовне выделов с ценными породами
        self._wood_background_cost = float(self.config.get('WOOD', 'background_cost'))

        # Колонка, в которой хранится тип основной породы выдела
        self.wood_forest_type_column = self.config.get('WOOD', 'forest_type_column')
        # Метки пород древесины (в том виде, в каком они хранятся в БД)
        wood_labels = self.config.get('WOOD', 'wood_labels')
        # Относительная стоимость древесины (в порядке перечисления меток)
        wood_cost = self.config.get('WOOD', 'wood_cost')
        wood_labels = wood_labels.split(',')
        wood_cost = [float(wc) for wc in wood_cost.split(',')]
        if len(wood_cost) != len(wood_labels):
            raise ConfigError('Wood types and Wood costs parameters have different lengths')
        self._specie_costs = [SpecieCost(l, c) for (l,c) in zip(wood_labels, wood_cost)]

        self.wood_cost_params = WoodCostParams(self._specie_costs, self.wood_perspect_factor, self._wood_background_cost)

        self.stocks = self.config.get('STOCKS', 'stocks')

        # Названия растровых слоев, описывающих дороги
        road_labels = self.config.get('TRANSPORT', 'road_labels')
        road_labels = road_labels.split(',')
        # Относительные стоимости движения
        road_costs = self.config.get('TRANSPORT', 'road_costs')
        road_costs = [float(rc) for rc in road_costs.split(',')]
        if len(road_costs) != len(road_labels):
            raise ConfigError('Road types and Road costs parameters have different lengths')
        self._road_costs = [RasterCost(l, c) for (l, c) in zip(road_labels, road_costs)]
        self.walking_cost_params = WalkingCostParams(self.stocks, self._road_costs)

        self.func_alpha = float(self.config.get('FUNCTION', 'alpha'))


    def get_wood_cost_model(self, grass, cumulative_cost_column):
        return WoodCost(grass, self.wood_name, self.wood_forest_type_column, self.wood_perspect_column, cumulative_cost_column)


