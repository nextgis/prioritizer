# encoding: utf-8

from ConfigParser import ConfigParser

from wood_cost import SpecieCost, WoodCostParams, WoodCost
from walking_cost import WalkingCostParams, WalkingCost, RasterCost

class ConfigError(Exception):
    pass

class ModelParams(object):
    def __init__(self, config_file):
        self.config = ConfigParser()
        self.config_file = config_file

        self.wood_name = None
        # Запас основной породы
        self.wood_perspect_column = None
        # Относительная значимость запаса
        self.wood_perspect_factor = None

        # Относительная стоимость древесины вовне выделов с ценными породами
        self._wood_background_cost = None

        # Колонка, в которой хранится тип основной породы выдела
        self.wood_forest_type_column = None
        # Метки пород древесины (в том виде, в каком они хранятся в БД)
        self.wood_labels = None
        # Относительная стоимость древесины (в порядке перечисления меток)
        self._specie_costs = None

        self.stocks = None

        # Названия растровых слоев, описывающих дороги
        self.road_labels = None
        # Относительные стоимости движения
        self._road_costs = None

        self.func_alpha = None

    @property
    def wood_labels_for_conf(self):
        return ','.join(self.wood_labels)
    @property
    def wood_cost_for_conf(self):
        return ','.join([str(s.wood_cost) for s in self._specie_costs])
    @property
    def road_labels_for_conf(self):
        return ','.join(self.road_labels)
    @property
    def road_costs_for_conf(self):
        return ','.join([str(r.WalkingCost) for r in self._road_costs])


    def load(self):
        self.config.readfp(open(self.config_file))

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
        wood_labels = wood_labels.split(',')
        self.wood_labels = wood_labels
        # Относительная стоимость древесины (в порядке перечисления меток)
        wood_cost = self.config.get('WOOD', 'wood_cost')
        wood_cost = [float(wc) for wc in wood_cost.split(',')]
        if len(wood_cost) != len(wood_labels):
            raise ConfigError('Wood types and Wood costs parameters have different lengths')
        self._specie_costs = [SpecieCost(l, c) for (l,c) in zip(wood_labels, wood_cost)]

        self.stocks = self.config.get('STOCKS', 'stocks')

        # Названия растровых слоев, описывающих дороги
        road_labels = self.config.get('TRANSPORT', 'road_labels')
        road_labels = road_labels.split(',')
        self.road_labels = road_labels
        # Относительные стоимости движения
        road_costs = self.config.get('TRANSPORT', 'road_costs')
        road_costs = [float(rc) for rc in road_costs.split(',')]
        if len(road_costs) != len(self.road_labels):
            raise ConfigError('Road types and Road costs parameters have different lengths')
        self._road_costs = [RasterCost(l, c) for (l, c) in zip(road_labels, road_costs)]

        self.func_alpha = float(self.config.get('FUNCTION', 'alpha'))

    @property
    def walking_cost_params(self):
        return WalkingCostParams(self.stocks, self._road_costs)
    @walking_cost_params.setter
    def walking_cost_params(self, value):
        self.stocks = value.stocks
        self._road_costs = value.costs_list
        self.road_labels = [rc.RasterName for rc in value.costs_list]

    @property
    def wood_cost_params(self):
        return WoodCostParams(self._specie_costs, self.wood_perspect_factor, self._wood_background_cost)
    @wood_cost_params.setter
    def wood_cost_params(self, value):
        self.wood_perspect_factor = value.persp_factor
        self._wood_background_cost = value.background_cost
        self.wood_labels = [wc.label for wc in value.woodcosts]
        self._specie_costs = value.woodcosts


    def save(self, filename):
        pattern = """
# Описание древесины
[WOOD]
# Имя векторного слоя, описывающего выдела
name = {param.wood_name}

# Запас основной породы
perspect_column = {param.wood_perspect_column}
# Относительная значимость запаса
perspect_factor = {param.wood_perspect_factor}

# Колонка, в которой хранится тип основной породы выдела
forest_type_column = {param.wood_forest_type_column}

# Метки пород древесины (в том виде, в каком они хранятся в БД)
wood_labels = {param.wood_labels_for_conf}
# Относительная стоимость древесины (в порядке перечисления меток)
wood_cost = {param.wood_cost_for_conf}

# Относительная стоимость древесины вовне выделов с ценными породами
background_cost = {param._wood_background_cost}


# Описание доставки
[TRANSPORT]
# Названия растровых слоев, описывающих дороги (порядок важен -- фон должен идти последним)
road_labels = {param.road_labels_for_conf}
# Относительные стоимости движения
road_costs = {param.road_costs_for_conf}

# Места приема/продажи древесины
[STOCKS]
# Имя растрового слоя, в котором хранятся точки приема
stocks = {param.stocks}

# Параметры функции
[FUNCTION]
# параметр сигмоидальной функции альфа
alpha = {param.func_alpha}
        """
        config = pattern.format(param=self)

        with open(filename, 'w') as config_file:
            config_file.write(config)

    def update_params(self, walk_cost_params, wood_cost_params, alpha):
        self.func_alpha = alpha
        self.wood_cost_params = wood_cost_params
        self.walking_cost_params = walk_cost_params


    def get_wood_cost_model(self, grass, cumulative_cost_column):
        return WoodCost(grass, self.wood_name, self.wood_forest_type_column, self.wood_perspect_column, cumulative_cost_column)