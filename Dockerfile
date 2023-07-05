# Menggunakan base image python:3.9-alpine
FROM python:3.9

# Set working directory di dalam container
WORKDIR /app

# Copy file requirements.txt ke dalam container
COPY requirements.txt .

# Install dependensi menggunakan pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode aplikasi ke dalam container
COPY . .

# Expose port yang digunakan oleh aplikasi
EXPOSE 5000

# Menjalankan perintah untuk menjalankan aplikasi saat container dijalankan
CMD ["python", "app.py"]
