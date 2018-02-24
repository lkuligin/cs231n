IP=$(wget -qO - http://ipecho.net/plain; echo)

gcloud compute --project=kuligin-sandbox firewall-rules create assignment1-rules --direction=INGRESS \
    --priority=1000 --network=default --action=ALLOW --rules=tcp:7000 --source-ranges=$IP
