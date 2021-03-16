FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-0
# If you're using a GPU enabled instance, use the image from below instead
#FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-3
WORKDIR /root

RUN pip install pandas numpy google-cloud-storage scikit-learn opencv-python Flask jsonpickle

RUN apt-get update; apt-get install git -y; apt-get install -y libgl1-mesa-dev

RUN git clone https://github.com/sergiovirahonda/AutomaticTraining-PredictionAPI.git

RUN mv /root/AutomaticTraining-PredictionAPI/data_utils.py /data_utils.py
RUN mv /root/AutomaticTraining-PredictionAPI/task.py /task.py
RUN mv /root/AutomaticTraining-PredictionAPI/email_notifications.py /email_notifications.py

EXPOSE 5000

ENTRYPOINT ["python","task.py"]
