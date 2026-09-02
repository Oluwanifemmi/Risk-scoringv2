#selecting version 3.9.0 for my python
FROM python:3.11-slim
#setting my working directory to app'''
WORKDIR /app
#copy the requirement to my working directory'''
COPY requirements.txt .
#using this methd , to make the dependcy install each time i run the command to save space'''
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
#copying the app to the working directory'''
COPY . /app
#setting the cmd to the fast api host and port values'''
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]