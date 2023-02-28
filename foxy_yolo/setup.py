from setuptools import setup
import glob

package_name = 'foxy_yolo'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/net_props', glob.glob('net_props/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='athykxz',
    maintainer_email='athykxz@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_node = foxy_yolo.yolo_node:main'
        ],
    },
)
