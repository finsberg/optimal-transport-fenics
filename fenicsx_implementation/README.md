# FEniCSx implementation

In this folder we also provide an implementation of the problem using FEniCSx (version 0.10.0).
To install FEniCSx, please follow the instructions on the [official FEniCSx website](https://fenicsproject.org/download/).

Once you have FEniCSx installed, you should install [dxa](https://github.com/jorgensd/dxa/) and [adios4dolfinx](https://github.com/jorgensd/adios4dolfinx) with
```bash
python3 -m pip install -r requirements.txt
```

If you are using docker for example you can run the following command to execute the code without installing anything on your system:

```bash
docker run --rm  -w /home/shared -v $PWD:/home/shared -it ghcr.io/fenics/dolfinx/dolfinx:v0.10.0 bash -c "python3 -m pip install -r requirements.txt && python3 optical_flow.py"
```
