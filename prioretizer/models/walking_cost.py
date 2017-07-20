# encoding: utf-8

from collections import namedtuple

import uuid

# Cost of walking for raster
RasterCost = namedtuple('RasterCost', 'RasterName, WalkingCost')


def temp_name(name, prefix):
    return 't' + prefix + name

class WalkingCost():
    def __init__(self, grass):
        """Create raster of walking cost using costs of particular rasters

        :param grass:   Initialized GIS engine to operate raster data
        :type grass:    prioretizer.grasslib.grasslib.GRASS
        """
        self.grs = grass

    def walking_cost(self, raster_costs, stocks, walking_cost, overwrite=False):
        """Create raster of walking cost using costs of particular rasters

        :param raster_costs: Walking costs for rasters
        :type raster_costs:  List of RasterCost tuples
        
        :param stocks: Name of raster for starting data points
        :param walking_cost: Name of output walking cost raster
        """
        prefix = uuid.uuid4().hex
        temp_cost = temp_name('temp_cost', prefix)
        road_list = []
        try:
            for rc in raster_costs:
                name, cost = rc
                output = temp_name(name, prefix)
                self.grs.grass.mapcalc("${out} = ${rast} * ${cost}", out=output, rast=name, cost=cost,
                                 quiet=True)
                road_list.append(output)

            self.grs.grass.run_command('r.patch', input=','.join(road_list), output=temp_cost)

            self.grs.grass.run_command(
                'r.cost',
                input=temp_cost, output=walking_cost,
                start_raster=stocks, overwrite=overwrite
            )

        finally:
            for name in road_list:
                self.grs.grass.run_command('g.remove', type='rast', name=name, flags='f')
            self.grs.grass.run_command('g.remove', type='rast', name=temp_cost, flags='f')

