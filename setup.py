from distutils.core import setup

setup(
    name='Deep Learning Colab Notebook Utils',
    version='0.1',
    packages=['dl_colab_notebooks', ],
    license='As is',
    install_requires=[
        'pydub',
        'librosa'
    ]
)
