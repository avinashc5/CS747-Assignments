#!/bin/bash

venv_name="env747"

read -p "Enter the path to Python 3 executable: " python_executable

# python_executable="path to python executable" # fill in your executable path here

# if [[ ! -f $python_executable ]]; then
if [[ ! $(type -P "$python_executable") ]]; then
  echo "\"$python_executable\" is not a valid executable. Please check the python_executable path used in the script."
  echo "Virtual environment '$venv_name' NOT created. Try again."
  exit
fi

$python_executable -m venv $venv_name

source $venv_name/bin/activate

pip install -r requirements.txt

echo "Virtual environment '$venv_name' created and libraries installed."




