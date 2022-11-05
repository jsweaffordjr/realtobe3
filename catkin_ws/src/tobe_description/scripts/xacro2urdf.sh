#!/bin/bash

rosrun xacro xacro --inorder `rospack find bioloid_typea_description`/typea.xacro -o `rospack find bioloid_typea_description`/typea.urdf &&
check_urdf `rospack find bioloid_typea_description`/typea.urdf && 
urdf_to_graphiz `rospack find bioloid_typea_description`/typea.urdf
