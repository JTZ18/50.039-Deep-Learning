#!/bin/bash

# Specify the URL of the file you want to download
url="http://example.com/path/to/checkpoints.zip"

# Use wget to download the file
wget $url

# Unzip the downloaded file
unzip checkpoints.zip

# Remove the zip file
rm checkpoints.zip