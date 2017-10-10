from setuptools import setup

setup(name='gridworlds',
      version='0.0.6',
      packages=["gridworlds", "gridworlds.envs"],
      package_data = {
          "gridworlds.envs": ["saved_maps/*.dat"]
      },
      install_requires=['gym']  # And any other dependencies we need
)  
