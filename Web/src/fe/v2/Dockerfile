FROM node:18-alpine

WORKDIR /app

# Copy package.json và package-lock.json trước để tận dụng cache
COPY package*.json ./
RUN npm install

# Copy các file khác
COPY . .

# Mở cổng 80 cho Vite dev server
EXPOSE 80

# Chạy Vite dev server với host 0.0.0.0 để có thể truy cập từ bên ngoài container
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "80"]