FROM jupyter/datascience-notebook:python-3.8.6

WORKDIR /app

COPY ./requirement.txt ./

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirement.txt
