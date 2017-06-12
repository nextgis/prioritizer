from setuptools import setup

setup(name='prioretizer',
      version='0.6.0',
      description='Prioretization',

      url='https://github.com/nextgis/prioritizer',

      author='NextGIS',
      author_email='',

      license='GPLv2+',

      packages=['prioretizer'],

      zip_safe=False,

      test_suite = 'nose.collector',
      tests_require = ['nose'],

      scripts=[
          "bin/apply_model",
          "bin/create_db",
          "bin/drop_layer",
          "bin/import_layer",
          "bin/list_layer",
          "bin/rasterize",
          "bin/train_model",
      ]
)
