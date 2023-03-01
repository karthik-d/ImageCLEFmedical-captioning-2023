#!/bin/bash

#Launch this script from this script's parent dir

current_dir="$(pwd)"
repo_folder_name="$(basename $(cd ../../..; pwd))"
evaluator_script_filename='evaluator.py'
docker_imagename="clefaicrowd/$repo_folder_name"
docker_containername=$repo_folder_name

[[ $1 = '--no-build' ]] && build=false || build=true

if [[ "$build" = true ]]; then
	docker build  -f ./Dockerfile -t $docker_imagename ../../../
else
	echo 'do not build docker image'
fi

docker run --rm --name $docker_containername -v "$current_dir/../../../../$repo_folder_name:/challenge" -w /challenge $docker_imagename python $evaluator_script_filename
