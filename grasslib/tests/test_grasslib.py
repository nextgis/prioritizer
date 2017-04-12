#!/bin/env python
# -*- coding: utf-8 -*-


import sys
import os

import shutil
import uuid

import tempfile

import unittest

from grasslib import GRASS
from grasslib import GrassRuntimeError


TEMPDIR = tempfile.gettempdir()

TEST_GISDB_NAME = uuid.uuid4().hex
TEST_LOCATION = uuid.uuid4().hex
DBASE = os.path.join(TEMPDIR, TEST_GISDB_NAME)

GRASS_LIB = '/usr/lib/grass70'
GRASS_EXEC = '/usr/bin/grass'


class TestGRASS(unittest.TestCase):

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
        try:
            grs.create_location_by_epsg(epsg_code=4326)
            self.assertTrue(os.path.isdir(dirname))

            self.assertRaises(
                GrassRuntimeError,
                grs.create_location_by_epsg, epsg_code=4326, drop_location=False
            )
        finally:
            shutil.rmtree(DBASE)

    def test_init_location(self):
        grs = GRASS(
            gisexec=GRASS_EXEC,
            gisbase=GRASS_LIB,
            grassdata=DBASE,
            location=TEST_LOCATION
        )
        try:
            grs.create_location_by_epsg(epsg_code=4326)

            grs.init_location_vars(grs.location, grs.mapset)
            region = grs.grass.region()
        finally:
            shutil.rmtree(DBASE)

        self.assertTrue(region.has_key('rows'))
        self.assertTrue(region.has_key('cols'))

if __name__ == '__main__':
    suite = unittest.makeSuite(TestGRASS, 'test')
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)