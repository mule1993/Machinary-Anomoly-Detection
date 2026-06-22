FROM astrocrpublic.azurecr.io/runtime:3.2-5
FROM quay.io/astronomer/astro-runtime:12.1.0-base

# Force pull the provider package directly into the Astro image layers
RUN pip install --no-cache-dir apache-airflow-providers-docker==3.14.0

USER root
RUN apt-get update && apt-get install -y docker.io

USER airflow