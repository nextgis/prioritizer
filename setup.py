from setuptools import setup

setup(name='prioretizer',
      version='1.0.0',
      description='Prioretization',

      url='https://github.com/nextgis/prioritizer',

      author='NextGIS',
      author_email='dmitrykolesov@nextgis.com',

      license='GPLv2+',

      packages=['prioretizer'],

      zip_safe=False,

      test_suite='nose.collector',
      tests_require=['nose'],

      scripts=[
          "bin/apply_model",
          "bin/create_db",
          "bin/drop_layer",
          "bin/export_layer",
          "bin/import_layer",
          "bin/list_layer",
          "bin/train_model",
          "bin/query_model"
      ]
)
