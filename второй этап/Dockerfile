FROM continuumio/anaconda3:latest

CMD mkdir /opt/results
WORKDIR /opt/results
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY final_solve.py solve.py
COPY lib.py lib.py
COPY final_solve/weights.pt weights.pt
COPY final_solve/user_publishers.npy user_publishers.npy
COPY final_solve/users_overall_active users_overall_active
COPY data/validate.tsv validate.tsv
CMD python solve.py weights.pt user_publishers.npy users_overall_active

