# Docker image
FROM python:3.7

# Package files
ADD packages /opt/gep_gbm

# Package install
RUN pip install -e /opt/gep_gbm/gb_api && pip install -e /opt/gep_gbm/gb_model

# App setup
WORKDIR /opt/gep_gbm/gb_api/gb_api
ENV FLASK_APP=run.py
RUN chmod +x run.sh

# Port
EXPOSE 5000

# Run
CMD ["bash", "./run.sh"] 

