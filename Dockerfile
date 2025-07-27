FROM python:3.10-slim

WORKDIR /app


COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

ENV GOOGLE_API_KEY=dQXvgctrytbukbvuwfgdt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]