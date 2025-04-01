# Components

# Preparation

Install uv

Run `cd components`

Run `uv sync`

Run `source .venv/bin//activate`

## Build, push and run

Now you can run `./build_component.sh`

## Virtual env for each component

```sh
cd <component>/src
uv pip install -r runtime-requirements.txt
uv lock -r runtime-requirements.txt
```
