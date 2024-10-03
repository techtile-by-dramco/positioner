# Techtile Scope

MAke sure to create a '.env' file in the main folder and include the Qualysis key:
`QUALYSIS_KEY="<PWD>"`

## Interface

```python
from Positioner import PositionerClient
import yaml

config = yaml.safe_load(("config.yml")
positioner = PositionerClient(config, backend="direct") # or backend="zmq"

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

or 

```sh
pip install git+https://github.com/techtile-by-dramco/positioner
```

## Update package

If it needs to be editable:
```sh
cd positioner
git pull
pip install --upgrade pip
pip install --editable .
```
or 

If you just want to use the lib:

```sh
pip install git+https://github.com/techtile-by-dramco/positioner -u
```

## Running example
```sh
cd positioner # if not already in this folder
cd examples
python .\example1.py
```
