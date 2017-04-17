# encoding: utf-8

from ConfigParser import ConfigParser

class Params:
    def __init__(self, config_file):
        self.config = ConfigParser()
        self.config.readfp(open(config_file))

        self.grass_lib = self.config.get('GRASS', 'grass_lib')
        self.grass_exec = self.config.get('GRASS', 'grass_exec')

        self.grassdata = self.config.get('DATABASE', 'grassdata')
        self.location = self.config.get('DATABASE', 'location')
        self.epsg = self.config.get('DATABASE', 'epsg')

        self.north = self.config.get('DATABASE', 'north')
        self.south = self.config.get('DATABASE', 'south')
        self.west = self.config.get('DATABASE', 'west')
        self.east = self.config.get('DATABASE', 'east')
        self.resolution = self.config.get('DATABASE', 'resolution')

