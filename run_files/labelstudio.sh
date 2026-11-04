#!/usr/bin/env bash
# Local Label Studio for the error viewer (tools/export_labelstudio.py).
# Bound to 127.0.0.1 only: the served clips are patient frames. Reach it with
#   ssh -L 8766:localhost:8766 ORSI-GPU2   ->  http://localhost:8766
# Login: user giorgio@orsi.local, password in $LS_DIR/password (both files 600, outside git).
set -euo pipefail

VIEWER=${VIEWER:-/home/gchie/workspace/viewer}
LS_DIR=$VIEWER/labelstudio
PORT=${PORT:-8766}
NAME=labelstudio-orsi-gchie
IMAGE=heartexlabs/label-studio:1.23.2

mkdir -p "$LS_DIR/data"
chmod 700 "$LS_DIR"
umask 077
[ -f "$LS_DIR/password" ] || python3 -c "import secrets; print(secrets.token_urlsafe(12))" > "$LS_DIR/password"
[ -f "$LS_DIR/token" ] || python3 -c "import secrets; print(secrets.token_hex(20))" > "$LS_DIR/token"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    docker start "$NAME" >/dev/null
else
    docker run -d --name "$NAME" --restart unless-stopped --user "$(id -u):$(id -g)" \
        -p "127.0.0.1:$PORT:8080" \
        -v "$LS_DIR/data:/label-studio/data" -v "$VIEWER:/label-studio/files:ro" \
        -e HOME=/label-studio/data \
        -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files \
        -e LABEL_STUDIO_USERNAME=giorgio@orsi.local -e LABEL_STUDIO_PASSWORD="$(cat "$LS_DIR/password")" \
        -e LABEL_STUDIO_USER_TOKEN="$(cat "$LS_DIR/token")" -e LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true \
        -e COLLECT_ANALYTICS=false -e LABEL_STUDIO_COLLECT_ANALYTICS=false -e SENTRY_DSN= -e FRONTEND_SENTRY_DSN= \
        "$IMAGE" >/dev/null
fi
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do sleep 3; done

# Label Studio >= 1.18 disables legacy API tokens by default; the exporter uses one.
docker exec "$NAME" bash -c 'cd /label-studio/label_studio && python manage.py shell -c "
from jwt_auth.models import JWTSettings
from organizations.models import Organization
for o in Organization.objects.all():
    s, _ = JWTSettings.objects.get_or_create(organization=o)
    s.legacy_api_tokens_enabled = True
    s.save()
"' >/dev/null 2>&1
echo "Label Studio up on http://127.0.0.1:$PORT"
