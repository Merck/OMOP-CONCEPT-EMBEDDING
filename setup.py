#!/usr/bin/env python3
import sys
import os
import numpy
import subprocess as sp
from glob import glob
from distutils.core import setup, Extension

# setup 3rd party headers for C++
sp.check_call('cd 3rd; if [ ! -d fmt ]; then ./update.py; fi', shell=True)

# download ehash
sp.check_call('if [ ! -f tools/ehash ]; then wget https://github.com/aaalgo/ehash/releases/download/v0.1/ehash -O tools/ehash; chmod +x tools/ehash ;fi', shell=True)

# build pcat
sp.check_call('cd tools; make pcat', shell=True)

NUMPY_INC = numpy.__file__.rsplit('/', 1)[0] + '/core/include'

omop_core = Extension('omop_core',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++17', '-I.', '-g', '-Wno-sign-compare', '-Wno-parentheses', '-Wno-narrowing', '-fopenmp', '-DDEBUG'], 
        libraries = ['gomp', 'z'],
        include_dirs = ['/usr/include', '/usr/local/include', NUMPY_INC,
            '3rd/pybind11/include',
            '3rd/xtl/include',
            '3rd/xtensor/include',
            '3rd/xtensor-python/include',
            '3rd/fmt/include',
            '3rd/spdlog/include',
            '3rd/zstr/src'],
        library_dirs = ['/usr/local/lib'],
        sources = ['omop_embed/omop_core.cpp']
        )

setup (name = 'omop_embed',
       version = '0.1',
       author = 'Dong Wei; Yupeng Li',
       author_email = '',
       license = 'Apache2.0',
       description = 'OMOP concept embedding training and patient-level outcome prediction',
       ext_modules = [omop_core],
       scripts = glob('scripts/omop_*')
       )


'''
Copyright © 2022 Merck & Co., Inc., Rahway, NJ, USA and its affiliates. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

