from setuptools import setup

setup(name='gridworlds',
      version='0.0.1',
      packages=["gridworlds", "gridworlds.envs"],
      package_data = {
          "gridworlds.envs": ["saved_maps/*.dat"]
      },
      install_requires=['gym']  # And any other dependencies foo needs
)  
