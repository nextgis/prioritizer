# encoding: utf-8

from collections import namedtuple


# This model is too complex
# SpecieCost = namedtuple('SpecieCost', 'label, wood_cost, d_cost, h_cost, b_cost')
# Use less parameters
SpecieCost = namedtuple('SpecieCost', 'label, wood_cost')
WoodCostParams = namedtuple('WoodCostParams', 'woodcosts, persp_factor, background_cost')


class WoodCost:
    def __init__(self,
                 grass,
                 mapname,
                 forest_type_column, forest_count_column,
                 diameter_column, height_column,
                 perspective_column, bonitet_column,
                 cumulative_cost
    ):
        """Create raster of walking cost using costs of particular rasters

        :param grass:   Initialized GIS engine to operate raster data
        :type grass:    prioretizer.grasslib.grasslib.GRASS
        """
        self.grs = grass

        self.mapname = mapname

        self.forest_type_column = forest_type_column
        self.forest_count_column = forest_count_column
        self.diameter_column = diameter_column
        self.height_column = height_column
        self.perspective_column = perspective_column
        self.bonitet_column = bonitet_column

        self.cumulative_cost = cumulative_cost

    def wood_cost(self, output, cost_params, overwrite=False):

        persp_factor = cost_params.persp_factor
        background_cost = cost_params.background_cost
        costs_list = cost_params.woodcosts

        # Create column with cost, then rasterize the column.
        # It should be faster and simple then raster calculations (less rasterisation)

        for wc in costs_list:
            # This model is too complex:
            # expression = "(%(count_col)s*%(wood_cost)s + %(bon_cost)s/%(bonitet)s + %(d_cost)s*%(diam)s + %(h_cost)s*%(h)s) * (%(persp_fact)s*%(persp)s+1)" \
                # % dict(
                    # count_col=self.forest_count_column,
                    # wood_cost=wc.wood_cost,
                    # bon_cost=wc.b_cost,
                    # bonitet=self.bonitet_column,
                    # diam=self.diameter_column,
                    # d_cost=wc.d_cost,
                    # h=self.height_column,
                    # h_cost=wc.h_cost,
                    # persp=self.perspective_column,
                    # persp_fact=persp_factor,
            # )
 
            # Use less paramethers:
            expression = "(%(wood_cost)s) * (%(persp_fact)s*%(persp)s+1)" \
                % dict(
                    wood_cost=wc.wood_cost,
                    persp=self.perspective_column,
                    persp_fact=persp_factor,
            )
            where = "%(type_col)s = \"%(label)s\"" % dict(type_col=self.forest_type_column, label=wc.label)
            self.grs.grass.run_command('v.db.update', map=self.mapname, column=self.cumulative_cost, value=expression, where=where)

        self.grs.grass.run_command('v.to.rast', input=self.mapname, output=output, use='attr', attribute_column=self.cumulative_cost, overwrite=overwrite)

        expression = '{result} = if(isnull({result}), {background}, {result})'.format(result=output, background=background_cost)
        self.grs.grass.run_command('r.mapcalc', expression=expression, overwrite=True)
