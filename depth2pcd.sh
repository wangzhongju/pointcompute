#!/usr/bin/env bash

for b in `ls ./data/depth/dp_2/ | awk -F '.' '{print $1}'`
do
	echo $b
	./build/point_compute -l=./data/image/img_2/$b.jpg -d=./data/depth/dp_2/$b.png -m=dp -i=./data/intrinsic/point_2.yaml
done
