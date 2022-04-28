FROM us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-7:latest
RUN pip install -U fire cloudml-hypertune scikit-learn==0.20.4 pandas==0.24.2 fire
WORKDIR /app
COPY train.py .

ENTRYPOINT ["./main.sh"]
