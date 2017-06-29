import os
import sys
import tensorflow as tf
from glob import glob
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

os.environ['OPT'] = ''

CFLAGS = os.environ.get('CFLAGS', '').split()
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

class CUDA_build_ext(build_ext):
    def build_extensions(self):
        _compiler = self.compiler

        def _(default_compile):
            def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
                if ext == '.cu':
                    default_compiler_so = _compiler.compiler_so
                    _compiler.compiler_so = ['nvcc']
                    extra_postargs = NVCCFLAGS
                    result = default_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
                    _compiler.compiler_so = default_compiler_so
                    return result

                return default_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
            return _compile

        def _object_filenames(source_filenames, strip_dir=0, output_dir=''):
            if output_dir is None:
                output_dir = '' 
            obj_names = [] 
            for src_name in source_filenames:
                base, ext = os.path.splitext(src_name)
                base = os.path.splitdrive(base)[1] # Chop off the drive
                base = base[os.path.isabs(base):]  # If abs, chop off leading /
                if ext not in _compiler.src_extensions:
                    raise UnknownFileError, \
                          "unknown file type '%s' (from '%s')" % (ext, src_name)
                if strip_dir:
                    base = os.path.basename(base)
                if ext != '.cu':
                    obj_names.append(os.path.join(output_dir, base + _compiler.obj_extension))
                else:
                    obj_names.append(os.path.join(output_dir, base + '_cu'+ _compiler.obj_extension))

            return obj_names

        self.compiler.src_extensions.append('.cu')
        self.compiler._compile = _(self.compiler._compile)
        self.compiler.object_filenames = _object_filenames
        build_ext.build_extensions(self)

setup(
    name='fast_tffm',
    packages=find_packages(),
    ext_modules=[
        Extension(
            "lib.libfast_tffm",
            glob('cc/*.cc') + glob('cc/*.cu'),
            language="c++",
            include_dirs=[tf.sysconfig.get_include()],
            extra_compile_args=CFLAGS,
        )
    ],
    cmdclass={'build_ext': CUDA_build_ext},
    scripts=['fast_tffm.py'],
)
