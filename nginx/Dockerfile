# Use an Nginx base image
FROM nginx

# Remove the default Nginx configuration file
RUN rm /etc/nginx/conf.d/default.conf

# Copy your own configuration file to the Nginx configuration directory
COPY nginx.conf /etc/nginx/conf.d/

# Expose the default HTTP and HTTPS ports
EXPOSE 80
EXPOSE 443