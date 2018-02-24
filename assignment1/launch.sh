NAME="assign-1"
STATUS=$(gcloud compute instances list --filter="name:$NAME" --format=json | python -c "import json,sys; print next(iter(json.load(sys.stdin)),{}).get('status', '')")

if [ "$STATUS" = "" ]
then
    gcloud compute addresses create "ipython" --region europe-west1

    echo "creating the VM"
    gcloud compute instances create "$NAME" \
    --zone "europe-west1-b" --machine-type "n1-standard-8" --subnet "default" \
    --no-restart-on-failure --maintenance-policy "TERMINATE" \
    --service-account "852282193923-compute@developer.gserviceaccount.com" \
    --scopes "https://www.googleapis.com/auth/devstorage.read_write","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring.write","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
    --image "ubuntu-1604-xenial-v20180126" --image-project "ubuntu-os-cloud" --boot-disk-size "200" --boot-disk-type "pd-standard" --boot-disk-device-name "$NAME"
elif [ "$STATUS" == "TERMINATED" ]
then
    echo "TERMINATED"
else 
    echo "status is $STATUS"
fi

gcloud compute ssh $NAME --zone=europe-west1-b
