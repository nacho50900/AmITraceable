# Build stage
FROM node:22-alpine AS builder
WORKDIR /app

# Install deps
COPY package.json package-lock.json ./
RUN npm ci --silent

# Copy sources
COPY . .

# Accept build-time API url. Vacío por defecto: la imagen incluye su
# propio nginx.conf que hace de proxy hacia el backend bajo el MISMO
# origen, así que el frontend puede (y debe) llamar a la API con rutas
# relativas -- ver src/api.ts. Solo tiene sentido pasar aquí una URL
# absoluta si sirves el frontend SIN este proxy delante (poco común).
ARG VITE_API_URL=""
RUN export VITE_API_URL=${VITE_API_URL} && npm run build

# Production image
FROM nginx:stable-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]