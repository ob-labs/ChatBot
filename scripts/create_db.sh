#!/usr/bin/env bash

source .env

if [[ $DB_NAME == "" ]]; then
    echo "Please provide a database name in the .env file"
    exit 1
fi

# Create the database
if [[ $DB_PASSWORD == "" ]]; then
    mysql -u $DB_USER -h $DB_HOST -P $DB_PORT -e "CREATE DATABASE $DB_NAME"
else
    mysql -u $DB_USER -p$DB_PASSWORD -h $DB_HOST -P $DB_PORT -e "CREATE DATABASE $DB_NAME"
fi

if [[ $? -ne 0 ]]; then
    echo "Failed to create database $DB_NAME"
    exit 1
fi
echo "Database $DB_NAME created successfully"