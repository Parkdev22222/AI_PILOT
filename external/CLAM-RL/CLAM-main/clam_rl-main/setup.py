from setuptools import setup, find_packages
from os import path

def _read_reqs(relpath):
    fullpath = path.join(path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))]

REQUIREMENTS = _read_reqs("requirements.txt")

setup(name='clam',
      version='1.0',
      description='Contrastive learning-based agent modeling for deep reinforcement learning',
      author='Wenhao Ma',
      author_email='mwh.uestc@gmail.com',
      packages=find_packages(),
      include_package_data=True,
      dependency_links=['https://github.com/openai/gym-recording.git@bea9968055b59551afe51357552fd3b00b65a839#egg=gym_recording'],
      zip_safe=False,
      install_requires=REQUIREMENTS,
)
