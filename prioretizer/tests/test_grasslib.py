#!/bin/env python
# -*- coding: utf-8 -*-


import sys
import os

import shutil
import uuid

import tempfile

import unittest


from ..grasslib.grasslib import GRASS
from ..grasslib.grasslib import GrassRuntimeError

from ..grasslib.configurator import Params

params = Params('prioretizer/tests/test_config.conf')

GRASS_LIB = params.grass_lib
GRASS_EXEC = params.grass_exec

TEST_LOCATION = uuid.uuid4().hex
DBASE = params.grassdata


class TestGRASS(unittest.TestCase):

    def tearDown(self):
        shutil.rmtree(DBASE, ignore_errors=True)

    def test_init(self):
        grs = GRASS(
            gisexec=GRASS_EXEC,
            gisbase=GRASS_LIB,
            grassdata=DBASE,
            location=TEST_LOCATION
        )
        assert hasattr(grs, 'grass')
        assert hasattr(grs, 'gsetup')
        assert hasattr(grs, 'garray')

        self.assertEqual(grs.location, TEST_LOCATION)
        self.assertEqual(grs.mapset, 'PERMANENT')
        self.assertEqual(grs.grassdata, DBASE)

    def test_create_location_by_epsg(self):
        grs = GRASS(
            gisexec=GRASS_EXEC,
            gisbase=GRASS_LIB,
            grassdata=DBASE,
            location=TEST_LOCATION
        )

        mapset = 'PERMANENT'
        dirname = os.path.join(DBASE, TEST_LOCATION, mapset)
        grs.create_location_by_epsg(epsg_code=4326)
        self.assertTrue(os.path.isdir(dirname))

        self.assertRaises(
            GrassRuntimeError,
            grs.create_location_by_epsg, epsg_code=4326, drop_location=False
        )

    def test_init_location(self):
        grs = GRASS(
            gisexec=GRASS_EXEC,
            gisbase=GRASS_LIB,
            grassdata=DBASE,
            location=TEST_LOCATION
        )
        grs.create_location_by_epsg(epsg_code=4326)

        grs.init_location_vars(grs.location, grs.mapset)
        region = grs.grass.region()

        self.assertTrue(region.has_key('rows'))
        self.assertTrue(region.has_key('cols'))

    def test_import_geofile(self):
        grs = GRASS(
            gisexec=GRASS_EXEC,
            gisbase=GRASS_LIB,
            grassdata=DBASE,
            location=TEST_LOCATION,
            init_loc=False
        )
        grs.create_location_by_epsg(epsg_code=32653)
        grs.init_location_vars(grs.location, grs.mapset)

        # grs.import_geofile('test3857.geojson', 'test', 'vect')
        # info = grs.grass.read_command('v.info', map='test')

        # REGION isn't defined yet
        self.assertRaises(
            Exception,
            grs.import_geofile,
            'prioretizer/tests/test_rast.tif', 'test', 'rast'
        )
        # Define region
        grs.grass.run_command(
            'g.region',
            n=params.north,
            s=params.south,
            w=params.west,
            e=params.east,
            res=params.resolution,
            flags='s'
        )
        grs.import_geofile('prioretizer/tests/test_rast.tif', 'test', 'rast')

        print "WARNING!!! Vector import isn't tested!"



if __name__ == '__main__':
    suite = unittest.makeSuite(TestGRASS, 'test')
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
