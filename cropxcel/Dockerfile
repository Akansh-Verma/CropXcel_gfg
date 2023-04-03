FROM continuumio/miniconda3:latest

ENV APP_HOME /app
WORKDIR $APP_HOME

# Removes output stream buffering, allowing for more efficient logging
ENV PYTHONUNBUFFERED 1

# Install conda dependencies
COPY environment.yml .
RUN conda env create --force -n base -f environment.yml
RUN echo "conda activate base" >> ~/.bashrc

# Copy local code to the container image.
COPY . .

# Activate conda environment and run the web service on container startup. 
CMD exec bash -c "source ~/.bashrc && gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 mysite.wsgi:application"
