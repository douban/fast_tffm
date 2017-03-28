import os
import sys
import tensorflow as tf
from glob import glob
from setuptools import setup, find_packages, Extension

os.environ['OPT'] = ''

CFLAGS = os.environ.get('CFLAGS', '').split()
CFLAGS.extend(['-fPIC', '-std=c++11'])
if sys.platform.startswith('linux'):
    CFLAGS.append('-D_GLIBCXX_USE_CXX11_ABI=0')
elif sys.platform.startswith('darwin'):
    CFLAGS.append('-undefined dynamic_lookup')


setup(
    name='fast_tffm',
    packages=find_packages(),
    ext_modules=[
        Extension(
            "lib.libfast_tffm",
            glob('cc/*.cc'),
            language="c++",
            include_dirs=[tf.sysconfig.get_include()],
            extra_compile_args=CFLAGS,
        )
    ],
    scripts=['fast_tffm.py'],
)
