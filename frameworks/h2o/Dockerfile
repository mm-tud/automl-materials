FROM python:3.7-slim-buster
RUN apt-get update && apt-get upgrade -y 
RUN apt-get install -y build-essential python3-pip python3-dev
RUN apt-get install -y default-jre

RUN pip3 -q install pip --upgrade
RUN pip3 install requests tabulate future
RUN pip3 uninstall h2o
RUN pip3 install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o

RUN pip3 install numpy pandas scikit-learn

RUN mkdir /data
RUN mkdir /workdir

WORKDIR /workdir

CMD python3 run.py
