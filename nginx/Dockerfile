FROM nginx:alpine

# Define build arguments
ARG APP_PORT
ARG HTTP_PORT

# Remove default nginx configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom nginx configuration template
COPY nginx.conf /etc/nginx/nginx.conf.template

# Copy and set up the entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Expose ports dynamically
EXPOSE ${HTTP_PORT}
EXPOSE ${APP_PORT}

# Start Nginx with our entrypoint script
ENTRYPOINT ["/docker-entrypoint.sh"] 