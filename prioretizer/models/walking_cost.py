# encoding: utf-8

import uuid

from collections import namedtuple

from utils import temp_name

# Cost of walking for raster
RasterCost = namedtuple('RasterCost', 'RasterName, WalkingCost')
WalkingCostParams = namedtuple('WalkingCostParams', 'stocks, costs_list')

class WalkingCost:
    def __init__(self, grass):
        """Create raster of walking cost using costs of particular rasters

        :param grass:   Initialized GIS engine to operate raster data
        :type grass:    prioretizer.grasslib.grasslib.GRASS
        """
        self.grs = grass

    def walking_cost(self, walking_cost, raster_costs_params, overwrite=False):
        """Create raster of walking cost using costs of particular rasters

        :param raster_costs: Walking costs for rasters
        :type raster_costs:  RasterCostParams
        
        :param stocks: Name of raster for starting data points
        :param walking_cost: Name of output walking cost raster
        """
        prefix = uuid.uuid4().hex
        temp_cost = temp_name('temp_cost', prefix)
        road_list = []

        raster_costs = raster_costs_params.costs_list
        stocks = raster_costs_params.stocks
        region = self.grs.grass.core.region()

        # NOTE: Cells must have approx square form!
        resolution = (region['nsres'] + region['ewres']) / 2

        try:
            for rc in raster_costs:
                name, cost = rc
                output = temp_name(name, prefix)
                self.grs.grass.mapcalc(
                    "${out} = ${rast} * ${cost} * ${resolution}",
                    out=output, rast=name, cost=format(cost, '.20f'), resolution=resolution,
                    quiet=True
                )
                road_list.append(output)

            self.grs.grass.run_command('r.patch', input=','.join(road_list), output=temp_cost)

            self.grs.grass.run_command(
                'r.cost',
                input=temp_cost, output=walking_cost,
                start_raster=stocks, overwrite=overwrite,
                quiet=True
            )

        finally:
            for name in road_list:
                self.grs.grass.run_command('g.remove', type='rast', name=name, flags='f', quiet=True)
            self.grs.grass.run_command('g.remove', type='rast', name=temp_cost, flags='f', quiet=True)
