import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'my_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ADD THIS LINE: Tells colcon to copy launch files to the install directory
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools', 'rclpy', 'my_interfaces'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='uditrana034@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'carla_node = my_pkg.CarlaNode:main',
            'ekf_node = my_pkg.ekf:main',
            'control_node = my_pkg.control:main',
            'logger_node = my_pkg.logger:main',
        ],
    },
)
