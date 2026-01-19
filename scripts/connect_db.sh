#!/usr/bin/env bash

source .env

# Connect to the database
if [[ $DB_PASSWORD == "" ]]; then
    mysql -u $DB_USER -h $DB_HOST -P $DB_PORT $DB_NAME
else
    mysql -u $DB_USER -p$DB_PASSWORD -h $DB_HOST -P $DB_PORT $DB_NAME
fi