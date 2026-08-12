FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.11.8

COPY docker/jupyter/requirements.txt /tmp/jupyter-requirements.txt

RUN python -m pip install --no-cache-dir -r /tmp/jupyter-requirements.txt
