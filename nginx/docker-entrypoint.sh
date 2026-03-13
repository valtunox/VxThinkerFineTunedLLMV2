#!/bin/sh
set -e

# Replace environment variables in nginx.conf
envsubst '${app_name} ${app_port}' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf

# Print configuration for debugging
echo "Nginx configuration:"
cat /etc/nginx/nginx.conf

# Start Nginx
exec nginx -g 'daemon off;' 