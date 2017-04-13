# encoding: utf-8

import os
import sys
import uuid

import shutil

import numpy as np

class GrassRuntimeError(Exception):
    pass

class GRASS:
    def __init__(self, gisexec, gisbase, grassdata, location, mapset="PERMANENT", init_loc=False):
        """Create wrapper around GRASS GIS
        :param gisexec:     Path to GRASS GIS executable file
        :param gisbase:     Path to GRASS GIS directory
        :param grassdata:   Path to GRASSDATA directory
        :param location:    Name of LOCATION
        :param mapset:      Name of MAPSET
        :param init_loc:    Run initialization procedure (set it True if the location exists)
        """
        self.gisexec = gisexec
        self.gisbase = gisbase
        self.grassdata = grassdata
        self.location = location
        self.mapset = mapset

        os.environ['GISBASE'] = gisbase
        sys.path.append(os.path.join(os.environ['GISBASE'], "etc", "python"))
        path = os.getenv('LD_LIBRARY_PATH')
        if path is None:
            os.environ['LD_LIBRARY_PATH'] = os.path.join(gisbase, 'lib')

        import grass.script as grass
        import grass.script.setup as gsetup

        self.grass = grass
        self.gsetup = gsetup

        import grass.script.array as garray
        import grass.pygrass.raster as grs_raster

        self.garray = garray
        self.rast = grs_raster

        if init_loc:
            self.init_location_vars(self.location, self.mapset)

    def init_location_vars(self, location, mapset):
        """Initialize location and set up connection
        """
        self.gsetup.init(self.gisbase,
                         self.grassdata, location, mapset)

    # def create_location(self, geofile, drop_location=True):
        # """
        # Create and activate new grass location using geofile.
        # :param geofile:     path to raster file
        # :param drop_location: Delete location if True(drop_location and the location exists)
        # """
        # # import ipdb; ipdb.set_trace()
        # path = os.path.join(self.grassdata, self.location)
#
        # if os.path.exists(path) and drop_location:
            # shutil.rmtree(path)
#
        # cmd = '%s -text -e -c %s %s' % (self.gisexec, geofile, path)
#
        # exitcode = os.system(cmd)
        # if exitcode != 0 and drop_location:
            # raise GrassRuntimeError("The command: %s. returns non-zero exit code" % (cmd, ))

    def create_location_by_epsg(self, epsg_code, drop_location=True):
        """
        Create location using EPSG code
        :param epsg_code:       EPSG code for new location.
        :param drop_location:   Delete location if True(drop_location and the location exists)

        """
        path = os.path.join(self.grassdata, self.location)

        if os.path.exists(path) and drop_location:
            shutil.rmtree(path)

        cmd = '%s -text -e -c EPSG:%s %s' % (self.gisexec, epsg_code, path)
        exitcode = os.system(cmd)

        if exitcode != 0:
            if drop_location:
                raise GrassRuntimeError("The command: %s. returns non-zero exit code" % (cmd, ))
            else:
                raise GrassRuntimeError("The location %s exists" % (self.location, ))

    def get_region_info(self):
        gregion = self.grass.region()
        return gregion
    
    def raster_to_array(self, map_name):
        """Считывает растр в текущем регионе и возвращает его в виде одной строки numpy.array
        """
                
        arr = self.garray.array()
        arr.read(map_name)
            
        return np.reshape(arr, -1)
    
    def rasters_to_array(self, maps):
        """Считывает список растров и возвращает их в виде двумерного numpy.array
        (каждый растр в отдельном столбце)
        """
        rows = self.get_region_info()['cells']
        cols = len(maps)
        
        # Наверное, есть способ узнать тип карты проще, чем этот
        rast = self.raster_to_array(maps[0])
        dtype = rast.dtype
        
        arr = np.empty((rows, cols), dtype)
    
        arr[:, 0] = rast
        for i in range(1, cols):
            rast = self.raster_to_array(maps[i])
            arr[:, i] = rast
            
        return arr
    
    def array_to_rast(self, arr, map_name, overwrite=None):
        """Сохраняет numpy.array в виде растра на диске
        """
        rast = self.garray.array()
        rast[...] = arr.reshape(rast.shape)
        rast.write(map_name, overwrite=overwrite)
        
    def copy_metadata_from_rast(self, copy_from, copy_to):
        """Копирует метаданные (r.support) из растра copy_from
        в растр copy_to.
        """
        # Временный файл: 
        tempfile = uuid.uuid4().hex

        try:
            self.grass.run_command('r.support', map=copy_from, savehistory=tempfile)
            self.grass.run_command('r.support', map=copy_to, loadhistory=tempfile)
        finally:
            os.unlink(tempfile)
                           
