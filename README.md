# Techtile Scope

## Interface

```python
from Positioner import PositionerClient
import yaml

config = yaml.safe_load(("config.yml")
positioner = PositionerClient(config, backend="direct")

positioner.start()
position = positioner.get_data()
positioner.stop()
```


## Installing package

Prior to installing ensure you have the latest pip version, e.g., `python3 -m pip install --upgrade pip`.

```sh
git clone https://github.com/techtile-by-dramco/positioner.git
cd positioner
pip install --editable .
```

## Update package

```sh
cd positioner
git pull
pip install --upgrade pip
pip install --editable .
```

## Running example
```sh
cd positioner # if not already in this folder
cd examples
python .\example1.py
```
