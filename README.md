# Techtile Scope

## Interface

```python
from TechtileScope import Scope

scope = Scope("192.108.0.251")
power_dBm = scope.get_power_dBm()
```


## Installing package

Prior to installing ensure you have the latest pip version, e.g., `python3 -m pip install --upgrade pip`.

```sh
git clone https://github.com/techtile-by-dramco/scope.git
cd scope
pip install --editable .
```

## Update package

```sh
cd scope
git pull
pip install --upgrade pip
pip install --editable .
```

## Running example
```sh
cd scope # if not already in this folder
cd examples
python .\example1.py
```
