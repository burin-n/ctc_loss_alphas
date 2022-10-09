from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='ctc_custom',
      ext_modules=[cpp_extension.CppExtension('ctc_custom', ['ctc_custom.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
