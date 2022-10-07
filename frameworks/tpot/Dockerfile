FROM python:3.7-slim-buster
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3-pip python3-dev
RUN pip3 -q install pip --upgrade
RUN pip3 install numpy scipy scikit-learn pandas joblib torch deap update_checker tqdm stopit xgboost 
RUN pip3 install tpot

RUN mkdir /data
RUN mkdir /workdir

WORKDIR /workdir

CMD python3 run.py
