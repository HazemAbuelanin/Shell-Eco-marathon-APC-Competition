## setup.py
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['shell_simulation'],
    package_dir={'': 'src'},
    requires=['rospy', 'std_msgs', 'nav_msgs', 'sensor_msgs', 'cv_bridge', 
              'visualization_msgs', 'geometry_msgs', 'tf', 'custom_msg']
)

setup(**setup_args)
