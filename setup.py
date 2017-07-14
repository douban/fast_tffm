import os
import re
import sys
import warnings
from glob import glob
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test
from distutils.errors import UnknownFileError

os.environ['OPT'] = ''

CFLAGS = os.environ.get('CFLAGS', '-O3').split()
CFLAGS.append('-std=c++11')
if sys.platform.startswith('linux'):
    CFLAGS.append('-D_GLIBCXX_USE_CXX11_ABI=0')
elif sys.platform.startswith('darwin'):
    CFLAGS.append('-undefined dynamic_lookup')


NVCCFLAGS = list(CFLAGS)
NVCCFLAGS.extend([
    '-Xcompiler', '-fPIC',
    '-gencode', 'arch=compute_35,code=compute_35',
    '-gencode', 'arch=compute_52,code=compute_52',
    '--expt-relaxed-constexpr'
])

CFLAGS.append('-fPIC')


def find_cuda():
    env = [
        'CUDA_TOOLKIT_ROOT',
        'CUDA_PATH'
    ]

    paths = [os.environ[e] for e in env if e in os.environ] + [
        '/opt/cuda',
        '/usr/local',
        '/usr/local/cuda',
    ]

    for p in paths:
        if os.path.isfile(os.path.join(p, 'bin', 'nvcc')):
            return p

CUDA_TOOLKIT_ROOT = find_cuda()
extra_link_args = []
sources = glob('cc/*.cc')

if CUDA_TOOLKIT_ROOT is not None:
    extra_link_args.extend([
        '-L', os.path.join(CUDA_TOOLKIT_ROOT, 'lib64'), '-l', 'cudart_static'
    ])
    sources.extend(glob('cc/*.cu'))
    CFLAGS.append('-DWITH_CUDA')
    NVCCFLAGS.append('-DWITH_CUDA')
else:
    warnings.warn('NO CUDA Found! Disable GPU kernels!', Warning)


def find_version(*paths):
    fname = os.path.join(*paths)
    with open(fname) as fhandler:
        version_file = fhandler.read()
        version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]',
                                  version_file, re.M)

    if not version_match:
        raise RuntimeError('Unable to find version string in %s' % (fname,))

    version = version_match.group(1)
    return version


class PyTest(test):

    def run_tests(self):
        import pytest
        sys.exit(pytest.main([]))


class TFBuild(build_ext):

    def finalize_options(self):
        build_ext.finalize_options(self)
        for d in self.get_tf_include():
            self.include_dirs.append(d)

    def get_tf_include(self):
        import tensorflow as tf
        include_dir = tf.sysconfig.get_include()
        protobuf_dir = os.path.join(
            os.path.dirname(os.path.dirname(include_dir)),
            'external', 'protobuf', 'src'
        )
        return [include_dir, protobuf_dir]

    def build_extensions(self):
        _compiler = self.compiler

        def _(default_compile):
            def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
                if ext == '.cu':
                    if CUDA_TOOLKIT_ROOT is None:
                        raise RuntimeError(
                            'Can not compile .cu files without CUDA!'
                        )

                    default_compiler_so = _compiler.compiler_so
                    _compiler.compiler_so = [
                        os.path.join(CUDA_TOOLKIT_ROOT, 'bin', 'nvcc')
                    ]
                    extra_postargs = NVCCFLAGS
                    result = default_compile(
                        obj, src, ext, cc_args, extra_postargs, pp_opts
                    )
                    _compiler.compiler_so = default_compiler_so
                    return result

                return default_compile(
                    obj, src, ext, cc_args, extra_postargs, pp_opts
                )
            return _compile

        def _object_filenames(source_filenames,
                              strip_dir=0, output_dir=''):
            if output_dir is None:
                output_dir = ''
            obj_names = []
            for src_name in source_filenames:
                base, ext = os.path.splitext(src_name)
                base = os.path.splitdrive(base)[1]  # Chop off the drive
                # If abs, chop off leading /
                base = base[os.path.isabs(base):]
                if ext not in _compiler.src_extensions:
                    raise UnknownFileError(
                        "unknown file type '%s' (from '%s')" %
                        (ext, src_name))
                if strip_dir:
                    base = os.path.basename(base)
                if ext != '.cu':
                    obj_names.append(
                        os.path.join(
                            output_dir,
                            base + _compiler.obj_extension
                        )
                    )
                else:
                    obj_names.append(
                        os.path.join(
                            output_dir,
                            base + '_cu' + _compiler.obj_extension
                        )
                    )

            return obj_names

        self.compiler.src_extensions.append('.cu')
        self.compiler._compile = _(self.compiler._compile)
        self.compiler.object_filenames = _object_filenames
        build_ext.build_extensions(self)


setup(
    name='fast_tffm',
    packages=find_packages(),
    version=find_version('tffm', '__init__.py'),
    zip_safe=False,
    ext_modules=[
        Extension(
            "lib.libfast_tffm",
            sources,
            language="c++",
            extra_compile_args=CFLAGS,
            extra_link_args=extra_link_args,
        )
    ],
    tests_require=['pytest-randomly', 'pytest'],
    cmdclass={
        'test': PyTest,
        'build_ext': TFBuild
    },
    scripts=['train.py']
)
