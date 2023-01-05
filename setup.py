from setuptools import setup, find_packages

requires = [
    'synapseclient',
    'numpy>=1.19.5',
    'bokeh<3.0.0',
    'svglib',
    'selenium',
    'natsort',
    'FlowKit',
    'reportlab',
    'statsmodels',
    'dataclasses',
    'matplotlib',
    'pandas',
    'pyarrow',
    'PyYAML',
    'scikit_learn',
    'scipy',
    'seaborn',
    'pyparsing',
    'webdriver-manager'
]

VERSION = '0.0.1'
DESCRIPTION = 'Systemic Lymphoid Architecture Response Assessment'
LONG_DESCRIPTION = '''

SYLARAS: Systemic Lymphoid Architecture Response Assessment

Sylaras is a tool to transform large single-cell datasets into information-rich
visual compendia detailing the time and tissue-dependent changes occurring in
immune cell frequency, function, and network-level architecture in response to
experimental perturbation.

'''
AUTHOR = 'Gregory J. Baker'
AUTHOR_EMAIL = 'gregory_baker2@hms.harvard.edu'
LICENSE = 'MIT License'
HOMEPAGE = 'https://github.com/gjbaker/sylaras'

setup(
    name='sylaras',
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    setup_requires=[],  # 'numpy==1.19.3'
    packages=find_packages(),
    install_requires=requires,
    entry_points={
        'console_scripts': [
            'sylaras=sylaras.scripts.sylaras:main',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: %s' % LICENSE,
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=HOMEPAGE,
    download_url='%s/archive/v%s.tar.gz' % (HOMEPAGE, VERSION),
    keywords='scripts single cell immunology data science',
    zip_safe=False,
)
