#!/bin/bash

# Make sure you have the latest version of the repo
echo
git pull
echo
echo setting up git

git config --global user.name nialldevlin 
git config --global user.email niallcdevlin@gmail.com
git remote set-url origin https://github.com/nialldevlin/CarND-Kidnapped-Vehicle-Project.git
echo

git remote -v

read -p "Commit Message: " commitmsg

git add .
git commit -m "$commitmsg"
git push origin master
