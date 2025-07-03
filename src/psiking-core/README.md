## Installation
### Build whl
* test env: python>=3.10, macOS 15.3.1 (arm)
* poetry is required

```
>> poetry build -f wheel                                  
Building psiking-core (0.0.1)
Building wheel
  - Building wheel
  - Built psiking_core-0.0.1-py3-none-any.whl
```

```
pip install dist/psiking_core-0.0.1-py3-none-any.whl --force-reinstall
```