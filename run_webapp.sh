# Reference: https://pyliaorachel.github.io/blog/tech/system/2017/07/07/flask-app-with-gunicorn-on-nginx-server-upon-aws-ec2-linux.html
# Reference: https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-centos-7
# Reference: https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-centos-7
gunicorn --bind 0.0.0.0:8080 --chdir app/ wsgi:app