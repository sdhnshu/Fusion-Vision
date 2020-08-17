#! /bin/sh

# Push fresh image to dockerhub
# docker build -t fusion-vision .
# docker tag fusion-vision sdhnshu/fusion-vision
# docker push sdhnshu/fusion-vision

# Deploy to openshift
oc delete project fusion-vision
oc new-project fusion-vision
oc new-app scripts/openshift.json