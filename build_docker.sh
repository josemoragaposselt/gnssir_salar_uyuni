#! /bin/bash

echo "Resetting docker"
echo *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
echo "Stop / Remove gnssrefl_jupyter container - command -> docker stop and docker rm gnssrefl_jupyter_docker"
echo *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
docker stop unavdocker/gnssrefl_jupyter
docker rm unavdocker/gnssrefl_jupyter

echo *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
echo "Remove python image, image that made gnssrefl_jupyter_docker: command -> docker image rm python"
echo *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
docker image rm unavdocker/gnssrefl_jupyter

echo *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
echo "docker run gnssrefl_jupyter unavdocker/container (pull image from dockerhub if not local)"
echo "Interactive, on port 8888, name = gnssrefl_jupyter"
echo "Mount external volumes - notebooks, orbits, working and bin directoy is copied into Docker container"
echo "* Starting Container..."
echo *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*

# Open jupyter notebook directly 
docker run \
        -it \
        -v $(pwd)/Files:/home/jovyan/gnssrefl_jupyter/Files \
        -p 8888:8888 \
        --name='gnssrefl_jupyter' \
        -e GRANT_SUDO=yes --user root \
        --restart=unless-stopped \
        unavdocker/gnssrefl_jupyter

# If you want to add your own external volume add another -v command to the docker run command
#-v ~/[path_to_local_directory]:/home/jovyan/gnssrefl_jupyter/[personal_directory]

echo *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*

# Do not open jupyter notebook directly - detached mode (-d), and then use docker execute to enter the container in a bash shell

#docker run -it -d -p 8888:8888 --name='gnssrefl_jupyter' -e GRANT_SUDO=yes gnssrefl_jupyter
#docker exec -it gnssrefl_jupyter /bin/bash
