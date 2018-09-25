FROM pytorch/pytorch:0.4_cuda9_cudnn7
FROM cuda90-tf160:latest

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install Tacotron dependencies
RUN pip install -r requirements.txt
RUN cd parallel_wavenet_vocoder && \
  pip install -e '.[train]' && \
  cd ..

RUN pip uninstall -q --yes torch && pip install torch==0.4.0

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python3", "app.py"]