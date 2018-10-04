FROM pytorch/pytorch:0.4_cuda9_cudnn7
FROM cuda90-tf160:latest

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install Tacotron dependencies
RUN pip3 install -r requirements.txt
RUN cd tacotron && \
  pip install -e '.[train]' && \
  cd ..
RUN cd parallel_wavenet_vocoder && \
  pip install -e '.[train]' && \
  cd ..

RUN pip3 uninstall -q --yes torch && pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
# CMD ["python3", "app.py"]