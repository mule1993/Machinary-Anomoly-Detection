FROM astrocrpublic.azurecr.io/runtime:3.2-5
FROM quay.io/astronomer/astro-runtime:12.1.0-base

USER root
RUN apt-get update && apt-get install -y docker.io

USER airflow