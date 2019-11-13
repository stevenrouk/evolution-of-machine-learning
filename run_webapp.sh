# Reference: https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-centos-7
gunicorn --bind 0.0.0.0:8080 --chdir app/ wsgi:app