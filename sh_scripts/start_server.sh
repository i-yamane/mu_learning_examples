set -eu

DB=$1
PORT=$2
#URI=mysql+pymysql://root:@localhost/$DB
URI=sqlite:///results/$DB
ART=results/${DB}_art/

# mysql.server start
# mysql -u root -e "create database $DB;" || echo "Assuming $DB already exists."

mlflow server \
    --backend-store-uri $URI \
    --default-artifact-root $ART \
    --host 0.0.0.0 \
    --port $PORT

