# Docker image
FROM python:3.7

# Working directory in container
WORKDIR /opt/gep_gbm

# API and model packages
COPY packages .

# Package dependencies
RUN pip install -e ./gb_api && pip install -e ./gb_model

# App setup
ENV FLASK_APP=./gb_api/gb_api/run.py
RUN chmod +x $FLASK_APP

# Port
EXPOSE 5000

# Run
# ENTRYPOINT bash
CMD python $FLASK_APP 

