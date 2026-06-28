# Use a lightweight stable Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /code

# Copy requirements text first to leverage Docker layer caching
COPY ./requirements.txt /code/requirements.txt

# Install dependencies using the CPU-only PyTorch setup from your requirements file
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your application code and weights into the container
COPY . .

# Expose port 7860 for Hugging Face routing
EXPOSE 7860

# Command to execute the app using python directly
CMD ["python", "app.py"]